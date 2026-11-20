"""Catálogo de benchmarks de slmbench y modos de evaluación.

Todos son benchmarks existentes y públicos. Se ejecutan con lighteval: los que lighteval trae de serie (IFEval, GSM8K, MATH-500)
y los que slmbench añade como tareas propias en `slmbench/benchmarks/` (IFStruct y los de programación; LiveCodeBench usa el prompt y
el ejecutor de lighteval con la métrica pass@1/pass@5 de slmbench). Los de programación (`passk`) generan varias respuestas por
problema para medir pass@1 y pass@5.
"""
from __future__ import annotations


GENERAL = {
    "ifeval": {
        "name": "IFEval", "spec": "ifeval|0", "hf": "google/IFEval", "n": 541, "metric": "prompt_level_strict_acc", "builtin": True,
        "link": "https://arxiv.org/abs/2311.07911", "requires": [],
        "desc_es": "Seguimiento de instrucciones verificables (longitud, formato, palabras obligatorias…). Mide si el modelo obedece, no lo que sabe.",
        "desc_en": "Verifiable instruction following (length, format, required words…). Measures whether the model obeys, not what it knows.",
    },
    "gsm8k": {
        "name": "GSM8K", "spec": "gsm8k|0", "hf": "openai/gsm8k:main", "n": 1319, "metric": "extractive_match", "builtin": True,
        "link": "https://arxiv.org/abs/2110.14168", "requires": [],
        "desc_es": "Problemas de matemáticas de primaria con enunciado: razonamiento aritmético en varios pasos, al alcance de un SLM.",
        "desc_en": "Grade-school math word problems: multi-step arithmetic reasoning within reach of a small model.",
    },
    "math_500": {
        "name": "MATH-500", "spec": "math_500|0", "hf": "HuggingFaceH4/MATH-500", "n": 500, "metric": "pass@k:k=1&n=1", "builtin": True,
        "link": "https://arxiv.org/abs/2305.20050", "requires": [],
        "desc_es": "500 problemas de matemáticas de competición (subconjunto de MATH). Más difícil: separa bien a los SLM que razonan.",
        "desc_en": "500 competition math problems (MATH subset). Harder: separates small models that reason well.",
    },
    "ifstruct": {
        "name": "IFStruct", "spec": "ifstruct|0", "hf": "LiquidAI/ifstruct-v1.0:default/test", "n": 2000, "metric": "pass_rate", "builtin": False,
        "link": "https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0", "requires": [],
        "desc_es": "Salida estructurada (Liquid AI): generar JSON o YAML que cumpla un esquema y todas las restricciones pedidas (número de "
                   "elementos, clave contenedora, bloque de código, sin comentarios, enums, rangos). Solo puntúa si se cumple todo.",
        "desc_en": "Structured output (Liquid AI): produce JSON or YAML that follows a schema and every requested constraint (item count, "
                   "wrapper key, code fence, no commentary, enums, ranges). A response scores only if everything holds.",
    },
}

EXTRA_GENERAL = {
    "mgsm_es": {
        "name": "MGSM (es)", "spec": "mgsm_es|0", "hf": "juletxara/mgsm:es/test", "n": 250, "metric": "accuracy", "builtin": False,
        "link": "https://arxiv.org/abs/2210.03057", "requires": [],
        "desc_es": "Problemas de matemáticas de primaria traducidos al español por personas (MGSM). Mide razonar en español; "
                   "puntúa el último número de la respuesta, como la evaluación oficial.",
        "desc_en": "Grade-school maths problems human-translated into Spanish (MGSM). Measures reasoning in Spanish; scores the "
                   "last number of the answer, like the official evaluation.",
    },
}

TOOLS = {
    f"bfcl_{c}": {
        "name": f"BFCL {label}", "spec": f"bfcl_{c}|0", "hf": f"bfcl:{c}", "n": n, "metric": "pass@1", "builtin": False,
        "prepare": "bfcl", "passk": True, "link": "https://gorilla.cs.berkeley.edu/leaderboard.html", "requires": [],
        "desc_es": f"Berkeley Function Calling Leaderboard, categoría {label.lower()}: {d_es} Llamada nativa a herramientas (tools / "
                   "tool_calls), corregida con el comprobador AST de BFCL.",
        "desc_en": f"Berkeley Function Calling Leaderboard, {label.lower()} category: {d_en} Native tool calling (tools / tool_calls), "
                   "judged with BFCL's AST checker.",
    }
    for c, label, n, d_es, d_en in (
        ("simple", "Simple", 400, "una función, una llamada.", "one function, one call."),
        ("multiple", "Multiple", 200, "elegir la función correcta entre 2–4.", "pick the right function out of 2–4."),
        ("parallel", "Parallel", 200, "varias llamadas a la misma función en una respuesta.", "several calls to the same function in one answer."),
        ("parallel_multiple", "Parallel multiple", 200, "varias llamadas a varias funciones.", "several calls to several functions."),
    )
}

LONG = {
    f"ruler_{k}": {
        "name": f"RULER NIAH {k}", "spec": f"ruler_niah_{k}|0", "hf": f"simonjegou/ruler:{length}/test", "n": 4000,
        "metric": "string_match_all", "builtin": False, "link": "https://arxiv.org/abs/2404.06654", "requires": [],
        "desc_es": f"Aguja en un pajar (RULER) con ~{k} tokens de contexto: encontrar uno o varios valores escondidos en un texto largo. "
                   "Mide si el modelo usa de verdad su contexto.",
        "desc_en": f"Needle in a haystack (RULER) with ~{k} tokens of context: find one or several values hidden in a long text. "
                   "Measures whether the model really uses its context.",
    }
    for k, length in (("4k", "4096"), ("8k", "8192"))
}

CODE = {
    "lcb": {
        "name": "LiveCodeBench", "spec": "livecodebench|0", "hf": "lighteval/code_generation_lite:v4_v5", "n": 268, "metric": "pass@1",
        "builtin": False, "passk": True, "min_tokens": 8192, "link": "https://arxiv.org/abs/2403.07974", "requires": [],
        "desc_es": "Problemas recientes de LeetCode/AtCoder/Codeforces (sin contaminación). Muy difícil para SLMs; descarga ≈1,6 GB.",
        "desc_en": "Recent LeetCode/AtCoder/Codeforces problems (contamination-free). Very hard for SLMs; ≈1.6 GB download.",
    },
}

AGENT = {
    f"swebench_lite{suffix}": {
        "name": f"SWE-bench Lite · {label}", "spec": f"swebench_lite{suffix}|0", "hf": "SWE-bench/SWE-bench_Lite:default/test", "n": 300,
        "metric": "resolved", "builtin": False, "harness": harness, "harness_name": label, "harness_note_es": note_es,
        "harness_note_en": note_en, "prepare": "agent", "link": "https://www.swebench.com/lite.html",
        "requires": [],
        "desc_es": f"Issues reales de GitHub en 12 proyectos de Python (django, sympy, pytest, requests…): el modelo, como agente con "
                   f"{label} ({d_es}), explora el repositorio, edita el código y ejecuta comandos hasta dar con el arreglo. Resuelta = "
                   "pasan los tests del arreglo real sin romper ninguno (harness oficial de SWE-bench). Todo dentro de un sandbox sin red.",
        "desc_en": f"Real GitHub issues from 12 Python projects (django, sympy, pytest, requests…): the model, as an agent with {label} "
                   f"({d_en}), explores the repository, edits the code and runs commands until it fixes the issue. Resolved = the tests "
                   "of the real fix pass and none breaks (official SWE-bench harness). Everything inside a sandbox without network.",
    }
    for suffix, harness, label, d_es, d_en, note_es, note_en in (
        ("", "mini-swe-agent", "mini-SWE-agent", "solo bash: un comando por respuesta, en texto, sin tool calls",
         "bash only: one command per answer, as text, no tool calls", "solo bash, sin tool calls", "bash only, no tool calls"),
        ("_pi", "pi", "Pi", "herramientas nativas read/write/edit/bash con tool calls", "native read/write/edit/bash tools via tool calls",
         "read/write/edit/bash con tool calls", "read/write/edit/bash via tool calls"),
        ("_opencode", "opencode", "OpenCode", "su agente build con sus herramientas (bash, read, edit, write, grep, glob…) por tool calls",
         "its build agent with its own tools (bash, read, edit, write, grep, glob…) via tool calls",
         "agente build, sus herramientas con tool calls", "build agent, its tools via tool calls"),
    )
}

MODES = {
    "general": {"es": "General", "en": "General", "benchmarks": list(GENERAL) + list(EXTRA_GENERAL),
                "desc_es": "Instrucciones, matemáticas (también en español) y salida estructurada.",
                "desc_en": "Instructions, maths (also in Spanish) and structured output."},
    "code": {"es": "Programación", "en": "Coding", "benchmarks": [],          # se completa con code.CATALOG
             "desc_es": "Solo programación: Python y otros lenguajes, generación y razonamiento sobre código.",
             "desc_en": "Coding only: Python and other languages, code generation and code reasoning."},
    "tools": {"es": "Herramientas", "en": "Tool calling", "benchmarks": list(TOOLS),
              "desc_es": "Llamada a funciones nativa (BFCL): elegir la función y los argumentos correctos.",
              "desc_en": "Native function calling (BFCL): picking the right function and arguments."},
    "long": {"es": "Contexto largo", "en": "Long context", "benchmarks": list(LONG),
             "desc_es": "Recuperar información de contextos largos (RULER, 4k y 8k tokens).",
             "desc_en": "Retrieving information from long contexts (RULER, 4k and 8k tokens)."},
    "agent": {"es": "Agente", "en": "Agent", "benchmarks": list(AGENT),
              "desc_es": "El modelo como agente de programación (SWE-bench Lite con mini-SWE-agent, Pi u OpenCode), en un sandbox sin red.",
              "desc_en": "The model as a coding agent (SWE-bench Lite with mini-SWE-agent, Pi or OpenCode), in a sandbox without network."},
}


def _code_catalog() -> dict:
    from .code_catalog import CATALOG    # solo datos: no importa lighteval (lento)
    return dict(CATALOG)


def catalog() -> dict[str, dict]:
    """Todos los benchmarks: clave -> ficha (con el modo al que pertenece)."""
    out = {}
    for k, v in {**GENERAL, **EXTRA_GENERAL}.items():
        out[k] = {**v, "mode": "general"}
    for k, v in TOOLS.items():
        out[k] = {**v, "mode": "tools"}
    for k, v in LONG.items():
        out[k] = {**v, "mode": "long"}
    for k, v in AGENT.items():
        out[k] = {**v, "mode": "agent"}
    code = {**_code_catalog(), **CODE}
    order = ["humaneval_plus", "mbpp_plus", "lcb", "cruxeval_o", "cruxeval_i"]
    for k in sorted(code, key=lambda k: (order.index(k) if k in order else len(order), k)):
        out[k] = {"builtin": False, **code[k], "mode": "code"}
    return out


def min_tokens(key: str) -> int:
    """Mínimo de tokens por respuesta que necesita el benchmark (p. ej. LiveCodeBench); 0 si basta el valor global."""
    return int(catalog().get(key, {}).get("min_tokens") or 0)


def effective_max_tokens(key: str, global_max: int) -> int:
    return max(global_max, min_tokens(key)) if global_max else global_max


def mode_benchmarks(mode: str) -> list[str]:
    cat = catalog()
    if mode == "all":                       # everything but the agent mode (hours per run, its own sandbox): it runs on its own
        return [k for k, v in cat.items() if v["mode"] != "agent"]
    return [k for k, v in cat.items() if v["mode"] == mode]


def spec_to_key(spec: str) -> str | None:
    base = spec.split("|")[0]
    for k, v in catalog().items():
        if v["spec"].split("|")[0] == base:
            return k
    return None


_sbx_missing: dict[str, list[str]] = {}
_sbx_lock = __import__("threading").Lock()       # the TUI checks the sandbox in the background at start-up: one check at a time


def missing_tools(key: str) -> list[str]:
    """What a benchmark lacks to run: every coding benchmark executes model code, and that only ever happens inside the
    slmbench sandbox, so it needs the sandbox (and, for MultiPL-E, the compiler inside it). Empty = ready."""
    v = catalog().get(key, {})
    req = v.get("requires", [])
    if v.get("prepare") == "agent":
        from .. import agent
        why = agent.status()
        return [why] if why else []
    if v.get("prepare") == "bfcl":
        from . import bfcl_data
        return [] if bfcl_data.prepared(v["hf"].split(":", 1)[1]) else ["dataset (slmbench setup --mode tools)"]
    if v.get("mode") != "code":
        return []
    from .. import sandbox
    with _sbx_lock:
        return _missing_in_sandbox(key, req, sandbox)


def _missing_in_sandbox(key: str, req: list, sandbox) -> list[str]:
    why = sandbox.status()
    if why:
        return [f"sandbox ({why})"]
    if req and key not in _sbx_missing:            # one sandbox call for every coding benchmark (each call takes ~0.5 s)
        code = {k: x.get("requires", []) for k, x in catalog().items() if x.get("mode") == "code" and x.get("requires")}
        absent = set(sandbox.missing_tools(sorted({t for r in code.values() for t in r})))
        for k, r in code.items():
            _sbx_missing[k] = [t for t in r if t in absent]
    return _sbx_missing.get(key, [])


def describe(key: str, lang: str = "en") -> str:
    v = catalog()[key]
    return v.get(f"desc_{lang}") or v.get("desc_es", "")
