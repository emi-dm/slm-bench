"""Motor de slmbench: ejecuta benchmarks existentes con lighteval contra un servidor con API OpenAI.

- lighteval aporta tareas y métricas; las que no trae (IFStruct, HumanEval+, MBPP+, CRUXEval, MultiPL-E, LiveCodeBench con pass@k) están en `slmbench/benchmarks/`
  y se cargan con `--custom-tasks lighteval_tasks.py`.
- El modelo es `lighteval_model.py` (`lighteval custom`): un cliente de /v1/chat/completions sin litellm ni cachés, que registra cada
  petición en un JSONL (velocidad, respuestas vacías o truncadas, progreso).
- Cada ejecución se anota en ~/.config/slmbench/active/ mientras dura, para que la TUI avise de ejecuciones en marcha.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics as st
import subprocess
import sys
import queue
import signal
import threading
import time
from pathlib import Path

from .benchmarks import catalog as cat
from .config import Settings, config_dir
from .i18n import T
from .servers import clean_model_id

LIGHTEVAL_VERSION = "0.13.0"
MODEL_FILE = Path(__file__).resolve().parent / "lighteval_model.py"
TASKS_FILE = Path(__file__).resolve().parent / "lighteval_tasks.py"


# ---------------------------------------------------------------------------------------------
# Instalación de lighteval (dependencia del proyecto; entorno aislado solo como último recurso)
# ---------------------------------------------------------------------------------------------
def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(os.environ.get("SLMBENCH_DATA") or Path(base) / "slmbench")


def lighteval_cmd() -> list[str]:
    """lighteval runs in this same Python (it's a dependency of slmbench): works with pip, pip --user, pipx, uv tool or a
    venv, wherever the `lighteval` script landed."""
    return [sys.executable, "-m", "lighteval"]


def env_python() -> Path:
    return Path(sys.executable)


def lighteval_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("lighteval") is not None


def installed_version() -> str | None:
    try:
        import importlib.metadata as md
        return md.version("lighteval")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------
# Selección de benchmarks
# ---------------------------------------------------------------------------------------------
def specs_for(keys: list[str], extra: str = "") -> list[str]:
    c = cat.catalog()
    specs = [c[k]["spec"] for k in keys if k in c]
    for t in [x.strip() for x in (extra or "").split(",") if x.strip()]:
        specs.append(t if "|" in t else f"{t}|0")
    return specs


def headline(task: str, metrics: dict):
    """Métrica principal de una tarea de lighteval (según el catálogo) -> (nombre, {value, stderr})."""
    key = cat.spec_to_key(task)
    want = cat.catalog()[key]["metric"] if key else None
    if want and want in metrics:
        return want, metrics[want]
    for m in ("pass@1", "prompt_level_strict_acc", "extractive_match", "value_accuracy"):
        if m in metrics:
            return m, metrics[m]
    if metrics:
        k = next(iter(metrics))
        return k, metrics[k]
    return None, None


# ---------------------------------------------------------------------------------------------
# Ejecuciones en marcha
# ---------------------------------------------------------------------------------------------
def _active_dir() -> Path:
    d = config_dir() / "active"
    d.mkdir(parents=True, exist_ok=True)
    return d


def active_runs() -> list[dict]:
    """Ejecuciones de slmbench en marcha (en este u otro proceso), con su progreso."""
    out = []
    d = config_dir() / "active"
    if not d.is_dir():
        return out
    for f in d.glob("*.json"):
        try:
            a = json.loads(f.read_text())
            os.kill(a["pid"], 0)
        except Exception:
            try:
                f.unlink()
            except OSError:
                pass
            continue
        done = 0
        try:
            with open(a["stats"]) as fh:
                done = sum(1 for _ in fh)
        except OSError:
            pass
        a["done"] = done
        out.append(a)
    return sorted(out, key=lambda a: a.get("started", 0))


def _expected_requests(specs: list[str], max_samples: int, passk_n: int = 1) -> int | None:
    c = cat.catalog()
    total = 0
    for s in specs:
        k = cat.spec_to_key(s)
        if not k:
            return None
        n = c[k]["n"]
        total += (min(n, max_samples) if max_samples else n) * (passk_n if c[k].get("passk") else 1)
    return total


# ---------------------------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------------------------
def model_env(s: Settings, model: str, stats: Path | None = None) -> dict:
    return {
        "SLMBENCH_LD_BASE_URL": s.base_url, "SLMBENCH_LD_MODEL": model,
        "SLMBENCH_LD_API_KEY": s.api_key or os.environ.get("SLMBENCH_API_KEY", ""),
        "SLMBENCH_LD_TEMPERATURE": str(s.temperature), "SLMBENCH_LD_MAX_TOKENS": str(s.ld_max_new_tokens or 2048),
        "SLMBENCH_LD_CONCURRENCY": str(max(1, s.workers)), "SLMBENCH_LD_TIMEOUT": str(int(s.timeout)),
        "SLMBENCH_LD_MAX_CONTEXT": str(s.max_context), "SLMBENCH_LD_EXTRA_BODY": s.extra_body or "",
        "SLMBENCH_LD_REASONING_BUDGET": str(s.reasoning_budget), "SLMBENCH_LD_STATS": str(stats or ""),
        "SLMBENCH_PASSK_N": str(max(1, s.passk_n)), "SLMBENCH_LD_SAMPLE_TEMPERATURE": str(s.passk_temperature),
        # mínimo de tokens por tarea (p. ej. LiveCodeBench necesita más que el valor global): {nombre de tarea: tokens}
        "SLMBENCH_LD_TASK_MIN_TOKENS": json.dumps({v["spec"].split("|")[0]: v["min_tokens"] for v in cat.catalog().values() if v.get("min_tokens")}),
    }


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(line: str) -> str:
    return _ANSI.sub("", line).strip()


def server_kind(base_url: str, served: dict | None) -> str:
    """"llama.cpp" when the server is llama.cpp (it reports its build) or runs on this machine: those get llama.cpp's own
    parameters (thinking_budget_tokens). A hosted API gets only standard OpenAI parameters."""
    from urllib.parse import urlparse
    if str((served or {}).get("backend", "")).startswith("llama.cpp"):
        return "llama.cpp"
    host = (urlparse(base_url).hostname or "").lower()
    return "llama.cpp" if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local") else "api"


def resolve_model(s: Settings, model: str, on_line=print) -> str:
    """The id the run is recorded under must be the model that answers. A single-model server (llama-server) answers with
    whatever it has loaded, whatever the request says: if the configured id isn't the one it serves, use the served one.
    A multi-model server (Ollama, `llama serve`, vLLM…) routes by id, so an unknown id is an error."""
    from .servers import probe
    try:
        ids = probe(s.base_url, s.api_key, timeout=5)
    except Exception:  # noqa: BLE001 - no server: the run itself reports it
        return model
    if not ids or model in ids:
        return model
    if len(ids) == 1:
        on_line("⚠ " + T(f"El servidor sirve «{ids[0]}», no «{model}»: se evalúa y se guarda como «{ids[0]}».",
                         f"The server serves '{ids[0]}', not '{model}': evaluating and recording it as '{ids[0]}'."))
        return ids[0]
    raise RuntimeError(T(f"El servidor no sirve «{model}». Disponibles: {', '.join(ids)}", f"The server doesn't serve '{model}'. Available: {', '.join(ids)}"))


def stop_process(pid: int, grace: float = 5.0) -> None:
    """Stops a run started by run() (its own process group): SIGTERM, and SIGKILL if it is still alive after `grace` s."""
    def send(sig):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return False
        except OSError:                    # not a group leader (older run): just the process
            try:
                os.kill(pid, sig)
            except OSError:
                return False
        return True

    if not send(signal.SIGTERM):
        return
    end = time.time() + grace
    while time.time() < end:
        try:
            os.killpg(pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    send(signal.SIGKILL)


def run(s: Settings, model: str, specs: list[str], *, on_line=print, cancel: threading.Event | None = None,
        replay: list[Path] | None = None, replay_only: bool = False, origin: str | None = None, env_override: dict | None = None,
        server: dict | None = None) -> Path:
    """Ejecuta los benchmarks y devuelve la carpeta de salida (summary.json, report.md, requests.jsonl, responses.jsonl,
    run.json, salida de lighteval).

    replay: answers of previous runs (responses.jsonl) reused when the request matches (resume); replay_only: never call the
    model (rescore); origin: the run this one resumes/rescores; env_override: generation parameters to reuse exactly."""
    if not lighteval_available():
        raise RuntimeError(T("lighteval no está instalado: reinstala slmbench (en el repo: uv tool install --editable . o pip install -e .).", "lighteval is not installed: reinstall slmbench (in the repo: uv tool install --editable . or pip install -e .)."))
    if not specs:
        raise RuntimeError(T("No hay benchmarks seleccionados.", "No benchmarks selected."))
    check_safe(specs)
    served = {}
    if not replay_only:
        model = resolve_model(s, model, on_line)
        try:                                   # what is being measured, read now (the server could change during the run)
            from .servers import model_meta
            served = {k: v for k, v in model_meta(s.base_url, model, timeout=4).items() if v not in (None, "")}
        except Exception:  # noqa: BLE001
            served = {}
    custom_label = s.label if s.label and getattr(s, "label_model", "") in ("*", model) else ""
    label = custom_label or re.sub(r"[^A-Za-z0-9._-]+", "_", clean_model_id(model))
    kind = "rescore" if replay_only else ("resume" if replay else "")
    out = Path(s.out_dir).expanduser() / f"{label}{'-' + kind if kind else ''}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    stats = out / "requests.jsonl"
    menv = model_env(s, model, stats)
    if env_override:                                   # same generation parameters as the original run (same answer keys)
        keep = ("TEMPERATURE", "SAMPLE_TEMPERATURE", "MAX_TOKENS", "TASK_MIN_TOKENS", "EXTRA_BODY", "REASONING_BUDGET", "PASSK_N")
        menv.update({k: v for k, v in env_override.items() if any(k.endswith(x) for x in keep)})
    menv["SLMBENCH_LD_SERVER_KIND"] = server_kind(s.base_url, served)
    menv["SLMBENCH_LD_RESPONSES"] = str(out / "responses.jsonl")
    if replay:
        menv["SLMBENCH_LD_REPLAY"] = os.pathsep.join(str(p) for p in replay)
    if replay_only:
        menv["SLMBENCH_LD_REPLAY_ONLY"] = "1"
    import dataclasses
    (out / "run.json").write_text(json.dumps({"model": model, "specs": specs, "kind": kind or "run", "origin": origin,
                                              "settings": {k: v for k, v in dataclasses.asdict(s).items() if k != "api_key"}},
                                             indent=1, ensure_ascii=False))
    (out / "model.json").write_text(json.dumps({k: (v if k != "SLMBENCH_LD_API_KEY" else ("***" if v else "")) for k, v in menv.items()}, indent=1))
    cmd = [*lighteval_cmd(), "custom", label, str(MODEL_FILE), ",".join(specs), "--custom-tasks", str(TASKS_FILE),
           "--output-dir", str(out / "lighteval-out"), "--save-details"]
    if s.ld_max_samples:
        cmd += ["--max-samples", str(s.ld_max_samples)]
    log = open(out / "run.log", "a", buffering=1)          # salida completa: la lee la vista en vivo de la TUI (desde cualquier proceso)
    _on_line = on_line

    def on_line(text):
        try:
            log.write(text + "\n")
        except (OSError, ValueError):
            pass
        _on_line(text)

    on_line("$ " + " ".join(cmd))
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", NO_COLOR="1", **menv)
    # own session/process group: cancelling stops lighteval and everything it started (stop_process)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(out), bufsize=1,
                         start_new_session=True)
    active = _active_dir() / f"{p.pid}.json"
    active.write_text(json.dumps({"pid": p.pid, "model": model, "label": label, "specs": specs, "stats": str(stats), "out": str(out),
                                  "total": _expected_requests(specs, s.ld_max_samples, s.passk_n), "started": time.time(),
                                  "per_task": {sp: _expected_requests([sp], s.ld_max_samples, s.passk_n) for sp in specs},
                                  "base_url": s.base_url, "max_samples": s.ld_max_samples}))
    buf, cancelled, api_errors, api_notes = "", False, [], []
    # The pipe is read in a thread: a blocking read would wait until lighteval prints 256 characters, which can take
    # minutes while it waits for the model, and the cancel request would not be seen until then.
    chunks: queue.Queue = queue.Queue()

    def reader():
        try:
            while chunk := p.stdout.read(256):
                chunks.put(chunk)
        except (OSError, ValueError):
            pass
        chunks.put("")

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            try:
                if not cancelled and cancel is not None and cancel.is_set():
                    cancelled = True
                    on_line(T("Cancelando…", "Cancelling…"))
                    threading.Thread(target=stop_process, args=(p.pid,), daemon=True).start()
                try:
                    chunk = chunks.get(timeout=0.2)
                except queue.Empty:
                    continue
            except KeyboardInterrupt:          # Ctrl-C in `slmbench run`: the child is in its own session, stop it here
                if cancel is None:
                    cancel = threading.Event()
                cancel.set()
                continue
            if not chunk:
                break
            buf += chunk
            parts = re.split(r"[\r\n]", buf)
            buf = parts.pop()
            for part in parts:
                c = clean(part)
                if c.startswith("SLMBENCH_API_ERROR"):
                    api_errors.append(c[len("SLMBENCH_API_ERROR "):])
                if c.startswith("SLMBENCH_API_NOTE"):
                    api_notes.append(c[len("SLMBENCH_API_NOTE "):])
                if c and not ("%|" in c and "100%" not in c):
                    on_line(c)
        if buf.strip():
            on_line(clean(buf))
        rc = p.wait()
    finally:
        if p.poll() is None:                   # error or Ctrl-C while processing output: don't leave lighteval running
            stop_process(p.pid)
            p.wait()
        try:
            active.unlink()
        except OSError:
            pass
        log.close()
    meta = {"model": model, "base_url": s.base_url, "specs": specs, "max_samples": s.ld_max_samples,
            "date": dt.datetime.now().isoformat(timespec="seconds"), "label": label, "lighteval": installed_version(),
            "temperature": s.temperature, "reasoning_budget": s.reasoning_budget, "max_new_tokens": s.ld_max_new_tokens or 2048,
            "passk_n": max(1, s.passk_n), "passk_temperature": s.passk_temperature,
            "returncode": rc, "cancelled": cancelled, "api_errors": len(api_errors), "api_error_examples": api_errors[:3],
            "api_notes": api_notes, "server_kind": menv.get("SLMBENCH_LD_SERVER_KIND"),
            "kind": kind or "run", "origin": origin, "server": served or server or {}, "custom_label": custom_label,
            **request_stats(stats)}
    summary = summarize(out / "lighteval-out")
    meta["leaderboard"] = None
    if summary and not cancelled and s.leaderboard_auto:
        try:
            from . import leaderboard
            leaderboard.record(s, meta, summary)
            meta["leaderboard"] = str(leaderboard.paths(s)["md"])
        except Exception as e:  # noqa: BLE001 - el leaderboard nunca debe romper una ejecución
            meta["leaderboard_error"] = f"{type(e).__name__}: {e}"
    (out / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report(meta, summary))
    if rc != 0 and not summary and not cancelled:
        raise RuntimeError(T(f"lighteval terminó con código {rc} sin resultados; revisa el log.", f"lighteval exited with code {rc} and no results; check the log."))
    return out


def load_run(run_dir: str | Path) -> tuple[Settings, str, list[str], dict]:
    """(settings, model, specs, model.json env) of a past run, from its run.json and model.json."""
    import dataclasses
    d = Path(run_dir)
    if not (d / "run.json").exists() or not (d / "responses.jsonl").exists():
        raise RuntimeError(T(f"{d.name} no guardó sus respuestas (es anterior a esta función): no se puede repuntuar ni reanudar.",
                             f"{d.name} didn't save its answers (it predates this feature): it can't be rescored or resumed."))
    info = json.loads((d / "run.json").read_text())
    names = {f.name for f in dataclasses.fields(Settings)}
    s = Settings(**{k: v for k, v in info.get("settings", {}).items() if k in names})
    env = json.loads((d / "model.json").read_text()) if (d / "model.json").exists() else {}
    return s, info["model"], info["specs"], env


def _server_of(run_dir) -> dict:
    """The model data (quant, size…) recorded by the original run, so a rescore lands in the same leaderboard row."""
    try:
        return json.loads((Path(run_dir) / "summary.json").read_text())["meta"].get("server") or {}
    except (OSError, ValueError, KeyError):
        return {}


def rescore(run_dir: str | Path, current: Settings, *, on_line=print, cancel: threading.Event | None = None) -> Path:
    """Scores a past run again from its saved answers, without calling the model (e.g. after a scorer fix)."""
    s, model, specs, env = load_run(run_dir)
    s.out_dir, s.leaderboard_dir, s.leaderboard_auto = current.out_dir, current.leaderboard_dir, current.leaderboard_auto
    return run(s, model, specs, on_line=on_line, cancel=cancel, replay=[Path(run_dir) / "responses.jsonl"], replay_only=True,
               origin=Path(run_dir).name, env_override=env, server=_server_of(run_dir))


def resume(run_dir: str | Path, current: Settings, *, on_line=print, cancel: threading.Event | None = None) -> Path:
    """Continues an interrupted run: reuses its saved answers and asks the model only for the missing ones."""
    s, model, specs, env = load_run(run_dir)
    s.out_dir, s.leaderboard_dir, s.leaderboard_auto = current.out_dir, current.leaderboard_dir, current.leaderboard_auto
    s.base_url, s.api_key = current.base_url, current.api_key
    return run(s, model, specs, on_line=on_line, cancel=cancel, replay=[Path(run_dir) / "responses.jsonl"],
               origin=Path(run_dir).name, env_override=env, server=_server_of(run_dir))


def request_stats(path: Path) -> dict:
    rows = []
    try:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        pass
    tps = [r["tps"] for r in rows if r.get("tps")]
    pps = [r["pps"] for r in rows if r.get("pps") and (r.get("pt") or 0) > 500]
    by_mode: dict[str, dict] = {}                     # speed per evaluation mode (long-context prompts are much slower)
    c = cat.catalog()
    for r in rows:
        key = next((k for k in (cat.spec_to_key(p) for p in str(r.get("task") or "").split("|") if p) if k), None)
        if key and r.get("tps"):
            m = by_mode.setdefault(c[key]["mode"], {"tps": [], "pps": []})
            m["tps"].append(r["tps"])
            if r.get("pps") and (r.get("pt") or 0) > 500:
                m["pps"].append(r["pps"])
    return {"requests": len(rows), "reused": sum(1 for r in rows if r.get("cached")), "empty_responses": sum(1 for r in rows if r.get("empty")),
            "truncated": sum(1 for r in rows if r.get("finish") == "length"),
            "gen_tps": round(st.median(tps), 1) if tps else None, "prompt_tps": round(st.median(pps)) if pps else None,
            "speed_by_mode": {md: {"gen_tps": round(st.median(v["tps"]), 1), "prompt_tps": round(st.median(v["pps"])) if v["pps"] else None}
                              for md, v in by_mode.items()}}


def summarize(out_dir: Path) -> dict:
    """Último JSON de resultados de lighteval -> {tarea: {métrica: {value, stderr}}} (sin «all»)."""
    files = sorted(Path(out_dir).glob("results/**/results_*.json"))
    if not files:
        return {}
    res = json.loads(files[-1].read_text()).get("results", {})
    out = {}
    for task, metrics in res.items():
        if task == "all":
            continue
        out[task] = {k: {"value": v, "stderr": metrics.get(f"{k}_stderr")} for k, v in metrics.items()
                     if not k.endswith("_stderr") and isinstance(v, (int, float))}
    return out


# ---------------------------------------------------------------------------------------------
# Informes
# ---------------------------------------------------------------------------------------------
def pct(v):
    return "-" if v is None or v != v else f"{v * 100:.1f}%"      # v != v: NaN (p. ej. pass_yaml sin prompts YAML)


def report(meta: dict, summary: dict) -> str:
    c = cat.catalog()
    budget = T("sin límite", "unlimited") if meta.get("reasoning_budget", -1) < 0 else f"≤ {meta['reasoning_budget']} tokens"
    if meta.get("server_kind") == "api" and meta.get("reasoning_budget", -1) >= 0:
        budget = T("no enviado (API externa: el presupuesto es un parámetro de llama.cpp)",
                   "not sent (hosted API: the budget is a llama.cpp parameter)")
    L = [f"# {T('Informe', 'Report')}: {meta['label']}", "",
         f"- {T('Modelo', 'Model')}: `{meta['model']}`  ·  {T('Servidor', 'Server')}: `{meta['base_url']}`",
         f"- {T('Fecha', 'Date')}: {meta['date']}  ·  lighteval {meta.get('lighteval')}",
         f"- {T('Muestras por benchmark', 'Samples per benchmark')}: {meta['max_samples'] or T('todas', 'all')}  ·  "
         f"{T('temperatura', 'temperature')} {meta['temperature']}  ·  {T('máx. tokens', 'max tokens')} {meta['max_new_tokens']}"
         + "".join(f" ({c[k]['name']}: {cat.effective_max_tokens(k, meta['max_new_tokens'])})" for k in
                   dict.fromkeys(cat.spec_to_key(t) for t in summary) if k and cat.effective_max_tokens(k, meta['max_new_tokens']) != meta['max_new_tokens'])
         + "  ·  "
         f"{T('razonamiento', 'reasoning')}: {budget}"]
    if any((cat.spec_to_key(t) and c[cat.spec_to_key(t)].get("passk")) for t in summary):
        n = meta.get("passk_n", 1)
        L.append("- pass@k: " + (T(f"{n} respuestas por problema (programación y herramientas) a temperatura {meta.get('passk_temperature')} "
                                    "(pass@1 y pass@5 con el estimador insesgado de Chen et al. 2021)",
                                    f"{n} answers per problem (coding and tool calling) at temperature {meta.get('passk_temperature')} "
                                    "(pass@1 and pass@5 with the unbiased estimator of Chen et al. 2021)") if n > 1 else
                                  T("1 respuesta por problema (solo pass@1)", "1 answer per problem (pass@1 only)")))
    if meta.get("requests"):
        L.append(f"- {T('Peticiones', 'Requests')}: {meta['requests']}  ·  {T('vacías', 'empty')}: {meta.get('empty_responses', 0)}  ·  "
                 f"{T('truncadas por max tokens', 'truncated by max tokens')}: {meta.get('truncated', 0)}  ·  "
                 f"{T('generación', 'generation')}: {meta.get('gen_tps') or '-'} tok/s  ·  prefill: {meta.get('prompt_tps') or '-'} tok/s")
    if meta.get("kind") == "rescore":
        L.append("- " + T(f"Repuntuada a partir de `{meta.get('origin')}` con sus respuestas guardadas (sin llamar al modelo).",
                          f"Rescored from `{meta.get('origin')}` using its saved answers (no calls to the model)."))
    elif meta.get("kind") == "resume":
        L.append("- " + T(f"Reanudada a partir de `{meta.get('origin')}`: {meta.get('reused', 0)} respuestas reutilizadas.",
                          f"Resumed from `{meta.get('origin')}`: {meta.get('reused', 0)} answers reused."))
    if meta.get("max_samples"):
        L.append("- ⚠ " + T(f"Ejecución parcial (`--max-samples {meta['max_samples']}`): cifras orientativas, no comparables con la literatura.",
                            f"Partial run (`--max-samples {meta['max_samples']}`): indicative numbers, not comparable with published results."))
    if meta.get("cancelled"):
        L.append("- ⚠ " + T("Cancelada por el usuario.", "Cancelled by the user."))
    for note in meta.get("api_notes") or []:
        L.append("- ⚠ " + T("Parámetro adaptado al proveedor: ", "Parameter adapted to the provider: ") + f"`{note[:200]}`")
    if meta.get("api_errors"):
        L.append("- ⚠ " + T(f"**Peticiones fallidas: {meta['api_errors']}** (puntúan 0). Ejemplo: `{(meta.get('api_error_examples') or [''])[0][:120]}`",
                            f"**Failed requests: {meta['api_errors']}** (they score 0). Example: `{(meta.get('api_error_examples') or [''])[0][:120]}`"))
        L.append("  " + T("Todos los errores, agrupados: tecla `l` en Resultados o `slmbench log <ejecución>`.",
                          "All errors, grouped: key `l` in Results or `slmbench log <run>`."))
    L += ["", f"## {T('Resumen', 'Summary')}", "",
          f"| Benchmark | {T('Modo', 'Mode')} | {T('Métrica', 'Metric')} | {T('Valor', 'Score')} | ± | pass@5 |", "|---|---|---|---:|---:|---:|"]
    for task, metrics in summary.items():
        name, m = headline(task, metrics)
        key = cat.spec_to_key(task)
        mode = c[key]["mode"] if key else "-"
        p5 = metrics.get("pass@5")
        L.append(f"| {c[key]['name'] if key else task} | {mode} | {name or '-'} | {pct(m['value']) if m else '-'} | "
                 f"{pct(m['stderr']) if m and m.get('stderr') is not None else '-'} | {pct(p5['value']) if p5 else '-'} |")
    L += ["", f"## {T('Todas las métricas', 'All metrics')}", "", f"| {T('Tarea', 'Task')} | {T('Métrica', 'Metric')} | {T('Valor', 'Score')} | ± |", "|---|---|---:|---:|"]
    for task, metrics in summary.items():
        for k, m in metrics.items():
            L.append(f"| `{task}` | {k} | {pct(m['value'])} | {pct(m['stderr']) if m.get('stderr') is not None else '-'} |")
    L += ["", "> " + T("Las respuestas vacías suelen ser modelos que agotan los tokens razonando: sube «máx. tokens» o limita el razonamiento.",
                       "Empty answers usually come from models that run out of tokens while reasoning: raise max tokens or cap the reasoning budget."), ""]
    return "\n".join(L)


def errors_markdown(run_dir: str | Path, tail: int = 80) -> str:
    """What went wrong in a run, from its files: failed requests grouped by message (requests.jsonl), parameters adapted to
    the provider, sandbox warnings and lighteval errors (run.log), and the end of the log."""
    d = Path(run_dir).expanduser()
    rows = []
    try:
        rows = [json.loads(x) for x in (d / "requests.jsonl").read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        pass
    try:
        log = (d / "run.log").read_text(errors="replace").splitlines()
    except OSError:
        log = []
    try:
        meta = json.loads((d / "summary.json").read_text())["meta"]
    except (OSError, ValueError, KeyError):
        meta = {}
    errs = [r for r in rows if r.get("error")]
    L = [f"# {T('Errores y log', 'Errors and log')}: {d.name}", "",
         f"- {T('Peticiones', 'Requests')}: {len(rows)}  ·  {T('fallidas', 'failed')}: **{len(errs)}**  ·  "
         f"{T('vacías', 'empty')}: {sum(1 for r in rows if r.get('empty') and not r.get('error'))}  ·  "
         f"{T('truncadas', 'truncated')}: {sum(1 for r in rows if r.get('finish') == 'length')}"]
    if meta:
        L.append(f"- {T('Código de salida de lighteval', 'lighteval exit code')}: {meta.get('returncode')}"
                 + (T("  ·  cancelada", "  ·  cancelled") if meta.get("cancelled") else ""))
    elif (d / "run.json").exists():
        L.append("- " + T("Sin summary.json: la ejecución no terminó (se puede reanudar con `u` en Resultados).",
                          "No summary.json: the run didn't finish (resume it with `u` in Results)."))
    groups: dict[str, dict] = {}
    for r in errs:
        msg = re.sub(r"\s+", " ", str(r["error"]))[:300]
        g = groups.setdefault(msg, {"n": 0, "tasks": set()})
        g["n"] += 1
        if r.get("task"):
            g["tasks"].add(str(r["task"]).split("|")[0])
    L += ["", f"## {T('Peticiones fallidas', 'Failed requests')}", ""]
    if groups:
        L += [T("Cada una puntúa 0. Agrupadas por mensaje del servidor:", "Each one scores 0. Grouped by the server's message:"), ""]
        for msg, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            L += [f"- **{g['n']}×** {', '.join(sorted(g['tasks'])) or ''}", "", "````text", msg, "````", ""]
        hints = []
        low = " ".join(groups).lower()
        if "401" in low or "403" in low or "unauthorized" in low or "api key" in low:
            hints.append(T("401/403: revisa la clave de API (Modelo a evaluar → Clave de API).", "401/403: check the API key (Model to evaluate → API key)."))
        if "429" in low or "rate" in low:
            hints.append(T("429: límite de peticiones del proveedor; baja las peticiones en paralelo.", "429: the provider's rate limit; lower the parallel requests."))
        if "context" in low or "too long" in low or "maximum" in low:
            hints.append(T("Contexto: el prompt + máx. tokens no cabe; sube el contexto del servidor o baja máx. tokens.",
                           "Context: prompt + max tokens doesn't fit; raise the server context or lower max tokens."))
        if "timeout" in low or "timed out" in low:
            hints.append(T("Timeout: sube el timeout en Ajustes o baja las peticiones en paralelo.", "Timeout: raise the timeout in Settings or lower the parallel requests."))
        if "connection" in low or "refused" in low:
            hints.append(T("Sin conexión: ¿sigue en marcha el servidor?", "No connection: is the server still running?"))
        if hints:
            L += [f"**{T('Pistas', 'Hints')}**", ""] + [f"- {h}" for h in hints] + [""]
    else:
        L += [T("Ninguna.", "None."), ""]
    notes = meta.get("api_notes") or [x.split("SLMBENCH_API_NOTE ", 1)[1] for x in log if "SLMBENCH_API_NOTE " in x]
    if notes:
        L += [f"## {T('Parámetros adaptados al proveedor', 'Parameters adapted to the provider')}", ""] + [f"- {n}" for n in notes] + [""]
    sbx = sorted({x.split("SLMBENCH_SANDBOX_WARNING ", 1)[1] for x in log if "SLMBENCH_SANDBOX_WARNING " in x})
    if sbx:
        L += [f"## {T('Avisos del sandbox', 'Sandbox warnings')}", ""] + [f"- {w}" for w in sbx] + [""]
    problems, i = [], 0
    exc = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b")
    while i < len(log):                                  # lighteval / Python errors, with their traceback
        x = log[i]
        if x.startswith("Traceback"):
            j = i + 1
            while j < len(log) and j - i < 40 and not exc.match(log[j]):
                j += 1
            problems.append("\n".join(log[i:j + 1]))
            i = j + 1
        elif re.search(r"\b(ERROR|CRITICAL)\b", x) or exc.match(x):
            problems.append(x)
            i += 1
        else:
            i += 1
    if problems:
        L += [f"## {T('Errores de lighteval', 'lighteval errors')}", ""]
        for b in problems[-8:]:
            L += ["````text", b[-2500:], "````", ""]
    L += [f"## {T('Final del log', 'End of the log')} ({min(tail, len(log))}/{len(log)} {T('líneas', 'lines')})", "",
          "````text", "\n".join(log[-tail:]) or T("(vacío)", "(empty)"), "````", "",
          f"{T('Log completo', 'Full log')}: `{d / 'run.log'}`"]
    return "\n".join(L)


def list_runs(out_dir: str) -> list[dict]:
    runs = []
    root = Path(out_dir).expanduser()
    if not root.is_dir():
        return runs
    for d in sorted([p for p in root.rglob("summary.json")], reverse=True):
        try:
            data = json.loads(d.read_text())
        except Exception:
            continue
        heads = {}
        for task, metrics in data["results"].items():
            name, m = headline(task, metrics)
            if m:
                heads[task] = m["value"]
        meta = data["meta"]
        runs.append({"dir": str(d.parent), "label": meta.get("label", "?"), "date": meta.get("date", ""), "model": meta.get("model", "?"),
                     "heads": heads, "max_samples": meta.get("max_samples"), "gen_tps": meta.get("gen_tps"),
                     "quant": (meta.get("server") or {}).get("quant", ""), "custom_label": meta.get("custom_label", ""),
                     "kind": meta.get("kind", "run"), "incomplete": bool(meta.get("cancelled") or meta.get("returncode"))})
    done = {r["dir"] for r in runs}
    for rj in root.rglob("run.json"):                     # interrupted runs (no summary.json): listed so they can be resumed
        if str(rj.parent) in done or not (rj.parent / "responses.jsonl").exists():
            continue
        try:
            info = json.loads(rj.read_text())
        except ValueError:
            continue
        st = info.get("settings", {})
        runs.append({"dir": str(rj.parent), "label": st.get("label") or info.get("model", "?"), "model": info.get("model", "?"),
                     "date": dt.datetime.fromtimestamp(rj.stat().st_mtime).isoformat(timespec="seconds"), "heads": {},
                     "max_samples": st.get("ld_max_samples"), "gen_tps": None, "kind": info.get("kind", "run"), "incomplete": True})
    runs.sort(key=lambda r: r["date"], reverse=True)
    return runs


def compare(dirs: list[str]) -> str:
    """Tabla comparativa de varias ejecuciones (una columna por ejecución)."""
    runs = []
    for d in dirs:
        data = json.loads((Path(d) / "summary.json").read_text())
        runs.append(data)
    c = cat.catalog()
    labels = [f"{r['meta']['label']}" for r in runs]
    if len(set(labels)) < len(labels):
        labels = [f"{r['meta']['label']} ({r['meta']['date'][5:16].replace('T', ' ')})" for r in runs]
    tasks = []
    for r in runs:
        for t in r["results"]:
            if t not in tasks:
                tasks.append(t)
    L = [f"# {T('Comparación', 'Comparison')}", "", T("Programación y herramientas: pass@1 / pass@5.", "Coding and tool calling: pass@1 / pass@5."), "",
         "| Benchmark | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(runs)]
    for t in tasks:
        vals, p5s = [], []
        for r in runs:
            _, m = headline(t, r["results"].get(t, {})) if t in r["results"] else (None, None)
            vals.append(m["value"] if m else None)
            p5s.append((r["results"].get(t, {}).get("pass@5") or {}).get("value"))
        best = max((v for v in vals if v is not None), default=None)
        key = cat.spec_to_key(t)
        cells = [("**" + pct(v) + "**" if v is not None and v == best and len(runs) > 1 else pct(v)) + (f" / {pct(p)}" if p is not None else "")
                 for v, p in zip(vals, p5s)]
        L.append(f"| {c[key]['name'] if key else t} | " + " | ".join(cells) + " |")
    L += [f"| {T('Muestras', 'Samples')} | " + " | ".join(str(r["meta"].get("max_samples") or T("todas", "all")) for r in runs) + " |",
          "| tok/s | " + " | ".join(str(r["meta"].get("gen_tps") or "-") for r in runs) + " |", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------
# Preparación por modo
# ---------------------------------------------------------------------------------------------
def check_safe(specs: list[str]) -> None:
    """Model-generated code only ever runs inside the slmbench sandbox. Refuses (RuntimeError) to start a run that would
    execute code without it: coding benchmarks while the sandbox isn't ready, or lighteval's own LiveCodeBench tasks
    (`lcb:*`), which run the code on the host."""
    from . import sandbox
    host_exec = [sp for sp in specs if sp.split("|")[0].startswith("lcb:")]
    if host_exec:
        raise RuntimeError(T(f"{', '.join(host_exec)}: el LiveCodeBench de lighteval ejecuta el código del modelo en tu máquina. "
                             "Usa el benchmark `lcb` de slmbench, que es el mismo pero en el sandbox.",
                             f"{', '.join(host_exec)}: lighteval's LiveCodeBench runs the model's code on your machine. "
                             "Use slmbench's `lcb` benchmark: the same benchmark, run in the sandbox."))
    code = [cat.catalog()[k]["name"] for k in (cat.spec_to_key(sp) for sp in specs) if k and cat.catalog()[k]["mode"] == "code"]
    if code:
        why = sandbox.status(refresh=True)
        if why:
            raise RuntimeError(T(f"{', '.join(code)} ejecutan código generado por el modelo y solo se hace dentro del sandbox, "
                                 f"que no está listo: {why}. Prepáralo con `slmbench setup --mode code`.",
                                 f"{', '.join(code)} run model-generated code, which only ever happens inside the sandbox, "
                                 f"and it isn't ready: {why}. Set it up with `slmbench setup --mode code`."))


def setup(mode: str, on_line=print, download: bool = True) -> dict:
    """Comprueba requisitos (compiladores) y descarga los datasets de los benchmarks del modo. Devuelve un resumen."""
    keys = cat.mode_benchmarks(mode)
    c = cat.catalog()
    report_ = {"mode": mode, "benchmarks": keys, "missing_tools": {}, "downloaded": [], "failed": {}, "sandbox": None}
    on_line(T(f"Modo «{mode}»: {len(keys)} benchmarks", f"Mode '{mode}': {len(keys)} benchmarks"))
    if any(c[k]["mode"] == "code" for k in keys):              # model code runs only in the sandbox: create / prepare it
        from . import sandbox
        on_line(T("Sandbox para el código del modelo (Docker Sandboxes):", "Sandbox for model code (Docker Sandboxes):"))
        report_["sandbox"] = sandbox.provision(on_line=lambda l: on_line("  " + l))
        sandbox.status(refresh=True)
    for k in keys:
        miss = cat.missing_tools(k)
        if miss:
            report_["missing_tools"][k] = miss
            on_line(f"  ⚠ {c[k]['name']}: " + T("falta o no funciona:", "missing or not working:") + f" {', '.join(miss)}")
    if not lighteval_available():
        on_line(T("lighteval no está instalado: reinstala slmbench (en el repo: uv tool install --editable . o pip install -e .).", "lighteval is not installed: reinstall slmbench (in the repo: uv tool install --editable . or pip install -e .)."))
        return report_
    if download:
        if any(c[k].get("prepare") == "bfcl" for k in keys):     # BFCL: original files -> local dataset (exact JSON)
            from .benchmarks import bfcl_data as bfcl
            on_line(T("  preparando BFCL (ficheros originales)…", "  preparing BFCL (original files)…"))
            if bfcl.prepare(on_line=on_line):
                report_["downloaded"] += [k for k in keys if c[k].get("prepare") == "bfcl"]
        for k in keys:
            if c[k].get("prepare"):
                continue
            hf = c[k]["hf"]
            repo, _, rest = hf.partition(":")
            subset, _, split = rest.partition("/")
            on_line(T(f"  descargando {c[k]['name']} ({hf})…", f"  downloading {c[k]['name']} ({hf})…"))
            code = ("import sys, datasets\nrepo, subset, split = sys.argv[1], sys.argv[2] or None, sys.argv[3] or None\n"
                    "ds = datasets.load_dataset(repo, subset, split=split) if split else datasets.load_dataset(repo, subset)\nprint('ok', len(ds) if split else 'all')")
            r = subprocess.run([str(env_python()), "-c", code, repo, subset, split], capture_output=True, text=True, timeout=3600)
            if r.returncode == 0:
                report_["downloaded"].append(k)
                on_line("    ✔ " + r.stdout.strip().splitlines()[-1])
            else:
                report_["failed"][k] = (r.stderr.strip().splitlines() or ["?"])[-1][:200]
                on_line("    ✘ " + report_["failed"][k])
    return report_
