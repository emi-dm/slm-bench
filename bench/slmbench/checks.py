"""Utilidades de verificación: extracción de código/JSON, validador de JSON Schema mínimo y ejecución aislada de código."""
from __future__ import annotations

import json
import re

# ---------- extracción ----------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def code_blocks(text: str, lang: str | None = None) -> list[str]:
    out = []
    for l, body in _FENCE.findall(text):
        if lang is None or l.lower() in (lang, "py" if lang == "python" else lang, ""):
            out.append(body)
    return out


def extract_code(text: str, lang: str = "python") -> str:
    """Devuelve el bloque de código más largo; si no hay vallas, el texto completo."""
    blocks = code_blocks(text, lang)
    if blocks:
        return max(blocks, key=len)
    return text


def extract_json(text: str):
    """Intenta parsear JSON del texto: completo, dentro de una valla, o el primer {...}/[...] equilibrado."""
    t = text.strip()
    for cand in [t] + code_blocks(t, "json"):
        try:
            return json.loads(cand.strip())
        except Exception:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
            return obj
        except Exception:
            continue
    raise ValueError("no se encontró JSON válido")


def strict_json(text: str):
    """JSON sin prosa alrededor (permite una valla ```json si es lo único)."""
    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n?```", t, re.S)
    if m:
        t = m.group(1).strip()
    return json.loads(t)


# ---------- JSON Schema mínimo ----------

def validate(instance, schema, path="$") -> list[str]:
    """Valida un subconjunto de JSON Schema: type, enum, properties, required, additionalProperties,
    items, minItems, maxItems, minimum, maximum, minLength, maxLength, pattern. Devuelve lista de errores."""
    errs: list[str] = []
    t = schema.get("type")

    def is_type(v, tt):
        if tt == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if tt == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if tt == "boolean":
            return isinstance(v, bool)
        if tt == "null":
            return v is None
        return isinstance(v, {"object": dict, "array": list, "string": str}[tt])

    if t and not any(is_type(instance, tt) for tt in (t if isinstance(t, list) else [t])):
        return [f"{path}: se esperaba {t}, hay {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} no está en {schema['enum']}")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in instance:
                errs.append(f"{path}: falta '{r}'")
        if schema.get("additionalProperties") is False:
            for k in instance:
                if k not in props:
                    errs.append(f"{path}: propiedad extra '{k}'")
        for k, sub in props.items():
            if k in instance:
                errs += validate(instance[k], sub, f"{path}.{k}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append(f"{path}: menos de {schema['minItems']} elementos")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append(f"{path}: más de {schema['maxItems']} elementos")
        if "items" in schema:
            for i, el in enumerate(instance):
                errs += validate(el, schema["items"], f"{path}[{i}]")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: {instance} < {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: {instance} > {schema['maximum']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(f"{path}: demasiado corto")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errs.append(f"{path}: demasiado largo")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errs.append(f"{path}: no cumple /{schema['pattern']}/")
    return errs


# ---------- ejecución aislada de código ----------

def run_python(code: str, tests: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Runs `code` + `tests` as an isolated Python program (python -I, CPU/memory limits) inside the slmbench sandbox (see
    slmbench.sandbox: a Docker Sandbox microVM with no network, or the host only with --sandbox local).
    Returns (ok, output)."""
    from .sandbox import execute
    # CPU limit per program; memory is capped by the sandbox itself (the microVM's memory), so no RLIMIT_AS here: on Linux it
    # made some reference tests fail (MBPP+ 255 builds a list of >2 GB) where macOS silently ignored it.
    prelude = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (%d, %d))\n"
        % (int(timeout), int(timeout))
    )
    program = prelude + "\n" + code + "\n\n" + tests + "\n"
    ok, out = execute({"main.py": program}, [(["python", "-I", "main.py"], timeout + 2, "run")])
    return ok, out.strip()[-500:]
