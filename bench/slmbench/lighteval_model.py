"""Modelo personalizado para `lighteval custom`: llama directamente a /v1/chat/completions de cualquier servidor compatible con OpenAI
(llama-server, Ollama, LM Studio, vLLM…) sin litellm.

lighteval carga este fichero por ruta (`lighteval custom <nombre> <este_fichero> <tareas>`) dentro de SU entorno, así que solo puede usar
la librería estándar y lighteval (no importa nada de slmbench). La configuración llega por variables de entorno, que rellena
`engine.run`:

  SLMBENCH_LD_BASE_URL      URL base de la API (…/v1)
  SLMBENCH_LD_MODEL         id del modelo en el servidor
  SLMBENCH_LD_API_KEY       clave (opcional)
  SLMBENCH_LD_TEMPERATURE   temperatura (0 por defecto)
  SLMBENCH_LD_MAX_TOKENS    máx. de tokens por respuesta; sustituye al de la tarea (los modelos que razonan necesitan margen)
  SLMBENCH_LD_CONCURRENCY   peticiones simultáneas (≤ slots del servidor)
  SLMBENCH_LD_TIMEOUT       timeout por petición en segundos
  SLMBENCH_LD_MAX_CONTEXT   contexto del servidor (para max_length)
  SLMBENCH_LD_EXTRA_BODY    JSON añadido a cada petición (p. ej. chat_template_kwargs)
  SLMBENCH_LD_REASONING_BUDGET  tope de tokens de razonamiento por respuesta (-1 = sin límite, 0 = sin razonar)
  SLMBENCH_LD_RESPONSES     JSONL where every answer is saved (key, task, text, reasoning, finish…): re-scoring and the explorer
  SLMBENCH_LD_REPLAY        answers of previous runs (paths separated by os.pathsep): an answer whose key matches is reused
  SLMBENCH_LD_REPLAY_ONLY   "1": never call the model; answers not found are left empty (strict re-scoring)

No usa la caché de muestras de lighteval: cada ejecución consulta al modelo (si cambias de GGUF con el mismo alias, no se reutilizan
respuestas viejas). Los errores de API no se convierten en silencio en un 0 %: se reintentan y, si persisten, se escribe una línea
`SLMBENCH_API_ERROR …` en la salida, que slmbench cuenta y muestra en el informe.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc


class _NoCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """urllib keeps every header on a redirect, Authorization included, even to another host. Drop it then: the API key
    goes only to the server it was configured for."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoCrossHostAuth)


class _NoCache:
    """lighteval exige `model._cache`; slmbench no cachea respuestas (cada ejecución consulta al modelo)."""

    def _init_registry(self, registry):
        pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"SLMBENCH_LD_{name}", default)


class SlmbenchOpenAIModel(LightevalModel):
    DATASET_SPLITS = 1          # con una API no hace falta agrupar por longitud; así la concurrencia abarca todas las muestras

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.base_url = _env("BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
        self.model = _env("MODEL") or config.model_name
        self.api_key = _env("API_KEY")
        self.temperature = float(_env("TEMPERATURE", "0") or 0)
        # varias respuestas por problema (pass@k): hay que muestrear, así que se usa esta temperatura
        self.sample_temperature = float(_env("SAMPLE_TEMPERATURE", "0.6") or 0.6)
        self.max_tokens = int(_env("MAX_TOKENS", "0") or 0)
        try:                                                  # mínimo por tarea, p. ej. {"livecodebench": 8192}
            self.task_min_tokens = {k: int(v) for k, v in json.loads(_env("TASK_MIN_TOKENS", "{}") or "{}").items()}
        except ValueError:
            self.task_min_tokens = {}
        self.concurrency = max(1, int(_env("CONCURRENCY", "1") or 1))
        self.timeout = float(_env("TIMEOUT", "600") or 600)
        self._max_length = int(_env("MAX_CONTEXT", "32768") or 32768)
        self.extra_body = json.loads(_env("EXTRA_BODY") or "{}")
        self.reasoning_budget = int(_env("REASONING_BUDGET", "-1") or -1)
        # llama.cpp-only parameters (thinking_budget_tokens) go only to llama.cpp or local servers; hosted APIs may reject them
        self.llama_params = _env("SERVER_KIND", "llama.cpp") == "llama.cpp"
        self._adapt: dict[str, str | None] = {}         # parameter -> replacement name (None = drop), learnt from 400 errors
        self._adapt_lock = threading.Lock()
        self.retries = 2
        self._cache = _NoCache()
        self.stats_path = _env("STATS")          # JSONL con una línea por petición (velocidad y progreso para slmbench)
        self._stats_lock = threading.Lock()
        self.responses_path = _env("RESPONSES")
        self.replay_only = _env("REPLAY_ONLY") == "1"
        self.replay: dict[str, dict] = {}
        for path in filter(None, _env("REPLAY").split(os.pathsep)):
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row.get("key") and "error" not in row:
                            self.replay[row["key"]] = row
            except OSError:
                pass
        self.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None, system_prompt=None)

    # ---- propiedades que exige LightevalModel ----
    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        return self._max_length

    # ---- llamada a la API ----
    def _post(self, messages: list[dict], max_tokens: int | None, stop: list[str] | None, seed: int, task: str = "",
              temperature: float | None = None, tools: list | None = None) -> tuple[str, str | None, dict]:
        temp = self.temperature if temperature is None else temperature
        body = {"model": self.model, "messages": messages, "temperature": temp, "stream": False, "seed": seed}
        if tools:                                         # native tool calling (BFCL): the model answers with tool_calls
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        stop = [s for s in (stop or []) if s]
        if stop:
            body["stop"] = stop[:4]
        if self.reasoning_budget >= 0 and self.llama_params:
            body["thinking_budget_tokens"] = self.reasoning_budget
        body.update(self.extra_body)
        self._apply_adapt(body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last, adaptations = None, 0
        for attempt in range(self.retries + 1 + 4):
            if attempt - adaptations > self.retries:
                break
            try:
                t0 = time.time()
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers)
                with _OPENER.open(req, timeout=self.timeout) as r:
                    data = json.load(r)
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                if msg.get("tool_calls"):                     # returned to lighteval as text that the BFCL checker parses
                    calls = [{"name": (c.get("function") or {}).get("name", ""), "arguments": (c.get("function") or {}).get("arguments", "")}
                             for c in msg["tool_calls"]]
                    content = "SLMBENCH_TOOL_CALLS:" + json.dumps({"tool_calls": calls, "content": content}, ensure_ascii=False)
                self._record({"t": round(time.time() - t0, 3), "ct": (data.get("usage") or {}).get("completion_tokens"),
                              "pt": (data.get("usage") or {}).get("prompt_tokens"), "tps": (data.get("timings") or {}).get("predicted_per_second"),
                              "pps": (data.get("timings") or {}).get("prompt_per_second"), "finish": choice.get("finish_reason"), "empty": not content.strip(),
                              "task": task, "out": " ".join(content.split())[-240:]})   # para la vista en vivo de la TUI
                meta = {"finish": choice.get("finish_reason"), "ct": (data.get("usage") or {}).get("completion_tokens"),
                        "t": round(time.time() - t0, 3)}
                return content, msg.get("reasoning_content"), meta
            except urllib.error.HTTPError as e:
                text = self._redact(e.read().decode(errors="replace"))
                last = f"HTTP {e.code}: {text[:200]}"
                if e.code in (400, 422) and adaptations < 4 and self._learn(text, body):
                    adaptations += 1
                    self._apply_adapt(body)
                    continue                               # the provider rejected a parameter: retry without it
                if 400 <= e.code < 500 and e.code != 429:
                    break                                  # error de la petición: reintentar no sirve
            except Exception as e:                         # red, timeout, JSON roto
                last = self._redact(f"{type(e).__name__}: {e}")
            time.sleep(min(30, 2 ** (attempt - adaptations)))
        print(f"SLMBENCH_API_ERROR {last}", file=sys.stderr, flush=True)
        self._record({"error": last, "empty": True, "task": task})
        return "", None, {"error": last}

    # parameters a hosted API may refuse, and what to do: rename (max_tokens on OpenAI reasoning models) or drop
    ADAPTABLE = {"max_tokens": "max_completion_tokens", "thinking_budget_tokens": None, "temperature": None, "seed": None,
                 "stop": None}

    _KEYLIKE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|(?i:api[_-]?key)[\"':= ]+[A-Za-z0-9_\-.]{12,})")

    def _redact(self, text: str) -> str:
        """Server error messages are saved (requests.jsonl, the log, the report): never with the API key in them, even
        if the provider echoes it back."""
        if self.api_key:
            text = text.replace(self.api_key, "***")
        return self._KEYLIKE.sub("***", text)

    def _learn(self, error_text: str, body: dict) -> bool:
        """Reads a 400 error; if it rejects one of the adaptable parameters, remembers how to adapt it (for every later
        request too) and says so once in the run's output. Returns True if the request should be retried."""
        try:
            err = (json.loads(error_text).get("error") or {})
            err = err if isinstance(err, dict) else {"message": str(err)}
        except (ValueError, AttributeError):
            err = {"message": error_text}
        msg = str(err.get("message") or error_text)
        low = msg.lower()
        if not any(w in low for w in ("unsupported", "unrecognized", "not supported", "unknown", "extra inputs", "not permitted",
                                      "not allowed", "does not support", "only the default")):
            return False
        param = err.get("param") if err.get("param") in self.ADAPTABLE else None
        if not param:
            param = next((k for k in self.ADAPTABLE if re.search(rf"\b{k}\b", msg) and k in body), None)
        if not param or param not in body:
            return False
        with self._adapt_lock:
            if param in self._adapt:
                return True                                # another thread learnt it already
            # max_tokens is renamed only when the server asks for max_completion_tokens; never dropped (it caps the answer)
            if param == "max_tokens" and "max_completion_tokens" not in msg:
                return False
            new = self.ADAPTABLE[param]
            self._adapt[param] = new
        what = f"'{param}' -> '{new}'" if new else f"'{param}' removed"
        print(f"SLMBENCH_API_NOTE the server rejected a parameter, adapted for the whole run: {what} ({msg[:160]})", file=sys.stderr, flush=True)
        return True

    def _apply_adapt(self, body: dict):
        for old, new in list(self._adapt.items()):
            if old in body:
                v = body.pop(old)
                if new:
                    body[new] = v

    def _record(self, row: dict):
        if not self.stats_path:
            return
        with self._stats_lock, open(self.stats_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def problem_key(messages: list[dict]) -> str:
        """Identifies a problem by its prompt (the explorer joins answers to lighteval's details with it)."""
        return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]

    def request_key(self, messages, max_tokens, stop, i, temperature, tools=None) -> str:
        """Identifies one answer: same prompt and generation parameters -> same key (for --resume and rescore)."""
        spec = {"m": messages, "max": max_tokens, "stop": [x for x in (stop or []) if x], "i": i, "temp": temperature,
                "rb": self.reasoning_budget, "x": self.extra_body}
        if tools:
            spec["tools"] = tools
        return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]

    def _save_answer(self, row: dict):
        if not self.responses_path:
            return
        with self._stats_lock, open(self.responses_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _generate(self, args):
        """Una respuesta: (mensajes, máx. tokens, stop, tarea, índice de la muestra, nº de muestras del problema)."""
        messages, max_tokens, stop, task, i, n, tools = args
        temp = self.sample_temperature if n > 1 else self.temperature   # n > 1: muestreo; n = 1: la temperatura configurada
        key = self.request_key(messages, max_tokens, stop, i, temp, tools)
        base = {"key": key, "problem": self.problem_key(messages), "task": task, "i": i, "n": n}
        old = self.replay.get(key)
        if old is not None:                                   # reused answer (resume / rescore): no call to the model
            self._record({"cached": True, "task": task, "finish": old.get("finish"), "empty": not (old.get("text") or "").strip(),
                          "out": " ".join((old.get("text") or "").split())[-240:]})
            self._save_answer({**base, **{k: old.get(k) for k in ("text", "reasoning", "finish", "ct", "t")}, "cached": True})
            return old.get("text") or "", old.get("reasoning")
        if self.replay_only:
            self._record({"error": "not in the saved answers", "empty": True, "task": task})
            self._save_answer({**base, "error": "not in the saved answers", "text": ""})
            return "", None
        text, reasoning, meta = self._post(messages, max_tokens, stop, seed=i, task=task, temperature=temp, tools=tools)
        self._save_answer({**base, "text": text, "reasoning": reasoning, **meta})
        return text, reasoning

    def _messages(self, doc: Doc) -> list[dict]:
        """Mensajes de chat de la tarea; si la tarea define `specific["system_prompt"]` se envía como mensaje de sistema."""
        msgs = self.prompt_manager.prepare_prompt_api(doc)
        system = (doc.specific or {}).get("system_prompt") if isinstance(doc.specific, dict) else None
        if system and not (msgs and msgs[0].get("role") == "system"):
            msgs = [{"role": "system", "content": system}] + msgs
        return msgs

    # ---- interfaz de lighteval ----
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        """Genera las respuestas de todos los problemas. Cada problema usa SUS parámetros (nº de muestras para pass@k, tokens,
        stop), aunque en una misma ejecución se mezclen tareas distintas; todas las peticiones van al pool en paralelo."""
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []
        for split in dataset.splits_iterator():
            if len(split) == 0:
                continue
            msgs = [self._messages(doc) for doc in split]
            jobs, owner = [], []
            for d, doc in enumerate(split):
                n = max(1, int(doc.num_samples or 1))
                max_tokens = self.max_tokens or doc.generation_size
                base = next((p for p in str(doc.task_name or "").split("|") if p in self.task_min_tokens), None)
                if base and max_tokens:
                    max_tokens = max(max_tokens, self.task_min_tokens[base])
                tools = (doc.specific or {}).get("tools") if isinstance(doc.specific, dict) else None
                for i in range(n):
                    jobs.append((msgs[d], max_tokens, doc.stop_sequences, str(doc.task_name or ""), i, n, tools))
                    owner.append(d)
            with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                outs = list(ex.map(self._generate, jobs))
            texts = [[] for _ in split]
            reasonings = [[] for _ in split]
            for d, (t, r) in zip(owner, outs):
                texts[d].append(t or "")
                reasonings[d].append(r)
            for d in range(len(split)):
                results.append(ModelResponse(text=texts[d] or [""], reasonings=reasonings[d], input=msgs[d]))
        return dataset.get_original_order(results)

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError("slmbench: las tareas por verosimilitud (logprobs) no están soportadas con una API de chat; usa tareas generativas.")

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError("slmbench: perplejidad no soportada con una API de chat.")
