"""Memory estimates with hf-mem (https://github.com/alvarobartt/hf-mem): weights + KV cache + runtime overhead of a model,
read from its Safetensors / GGUF metadata (HTTP Range requests on the Hub, or the header of a local file) without
downloading the weights.

hf-mem runs in a subprocess (it is asyncio + httpx2; this keeps it out of curses and lets us time it out). Results are
cached per process, so the TUI can ask again on every redraw for free.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading

_cache: dict[tuple, dict] = {}
_pending: set[tuple] = set()
_lock = threading.Lock()

GB = 1024 ** 3


def command() -> list[str] | None:
    """How to run hf-mem: the installed package (a slmbench dependency), the `hf-mem` command, or `uvx hf-mem`."""
    try:
        import importlib.util
        if importlib.util.find_spec("hf_mem"):
            return [sys.executable, "-c", "from hf_mem.cli import main; main()"]
    except (ImportError, ValueError):
        pass
    if shutil.which("hf-mem"):
        return ["hf-mem"]
    if shutil.which("uvx"):
        return ["uvx", "hf-mem"]
    return None


def _args(model: str, ctx: int | None, parallel: int, file: str | None) -> list[str]:
    args = [model, "-o", "json"]
    if file:
        args += ["--file", file]
    if ctx:
        args += ["--max-model-len", str(int(ctx))]
    if parallel and parallel > 1:
        args += ["--batch-size", str(int(parallel))]
    return args


def estimate(model: str, ctx: int | None = None, parallel: int = 1, file: str | None = None, timeout: float = 90) -> dict:
    """hf-mem's JSON for `model`: a Hub id (`org/repo`, `org/repo:Q4_K_M` for a GGUF quant) or a local path (a .gguf file
    or a model folder). ctx = context length for the KV cache (None = the model's); parallel = concurrent sequences.
    Raises RuntimeError with hf-mem's own message when it fails."""
    model = os.path.expanduser(model.strip())
    if os.path.isfile(model) and not file:                 # a local file: hf-mem wants the folder + --file
        model, file = os.path.dirname(os.path.abspath(model)), os.path.basename(model)
    key = (model, ctx, parallel, file)
    if key in _cache:
        return _cache[key]
    cmd = command()
    if not cmd:
        raise RuntimeError("hf-mem is not installed (pip install hf-mem, or install uv to run it with uvx)")
    env = {**os.environ, "NO_COLOR": "1"}
    try:
        p = subprocess.run(cmd + _args(model, ctx, parallel, file), capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"hf-mem took more than {timeout:.0f} s (no connection to the Hub?)") from None
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip().splitlines()
        raise RuntimeError((msg[-1] if msg else f"hf-mem failed ({p.returncode})").removeprefix("hf-mem: error: "))
    try:
        data = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError("hf-mem returned something that is not JSON: " + p.stdout[:200]) from None
    _cache[key] = data
    return data


def estimate_async(model: str, ctx: int | None = None, parallel: int = 1, file: str | None = None,
                   fallbacks: tuple = ()) -> dict | None:
    """Non-blocking: the cached result, {"error": …} if it failed, or None while it is being computed in the background
    (the caller redraws later and asks again). fallbacks: other names to try, in order, if `model` fails (e.g. the
    GGUF repo first, then the original safetensors one). The result says which one answered in "_target"."""
    key = (os.path.expanduser(model.strip()), ctx, parallel, file, tuple(fallbacks))
    with _lock:
        if key in _errors:
            return {"error": _errors[key]}
        hit = _async_done.get(key)
        if hit is not None:
            return hit
        if key in _pending:
            return None
        _pending.add(key)

    def work():
        err = None
        try:
            for cand in (model, *fallbacks):
                try:
                    res = {**estimate(cand, ctx, parallel, file), "_target": cand}
                    with _lock:
                        _async_done[key] = res
                    return
                except Exception as e:  # noqa: BLE001
                    err = err or e                  # report the first (preferred) target's error
            with _lock:
                _errors[key] = str(err)
        finally:
            with _lock:
                _pending.discard(key)

    threading.Thread(target=work, daemon=True).start()
    return None


_async_done: dict[tuple, dict] = {}
_errors: dict[tuple, str] = {}


def system_memory() -> int:
    """Physical memory of this machine in bytes (0 if unknown). On Apple Silicon it is also the GPU's memory."""
    try:
        if platform.system() == "Darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout.strip())
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def gb(n: int | float | None) -> str:
    return "-" if n is None else f"{n / GB:.1f} GB" if n >= GB / 10 else f"{n / 1024 ** 2:.0f} MB"


def parts(d: dict) -> dict:
    """The numbers that matter, from one hf-mem estimate (single-file form)."""
    kv = d.get("kv_cache") or {}
    w = d.get("weights") or {}
    return {"total": d.get("total"), "weights": w.get("bytes"), "params": w.get("params"), "kv": kv.get("bytes"),
            "ctx": kv.get("max_model_len"), "batch": kv.get("batch_size"), "kv_dtype": kv.get("dtype"),
            "overhead": (d.get("overhead") or {}).get("bytes"), "format": d.get("format"), "file": d.get("file")}


def fit(total: int | None, ram: int | None = None) -> str:
    """'ok' | 'tight' | 'no' | '' against this machine's memory (the OS and other apps need some too)."""
    ram = system_memory() if ram is None else ram
    if not total or not ram:
        return ""
    return "ok" if total <= ram * 0.6 else "tight" if total <= ram * 0.85 else "no"


def summary(d: dict) -> str:
    """One line: ≈ 2.7 GB · weights 1.1 GB + KV 0.9 GB (8,192 ctx) + overhead 0.7 GB."""
    if d.get("error"):
        return "⚠ " + d["error"]
    if d.get("files"):
        tots = [f.get("total") or 0 for f in d["files"]]
        return f"{len(tots)} GGUF files · {gb(min(tots))} – {gb(max(tots))}"
    p = parts(d)
    bits = [f"weights {gb(p['weights'])}"]
    if p["kv"]:
        bits.append(f"KV {gb(p['kv'])} ({p['ctx']:,} ctx" + (f" × {p['batch']}" if (p["batch"] or 1) > 1 else "") + ")")
    if p["overhead"]:
        bits.append(f"overhead {gb(p['overhead'])}")
    return f"≈ {gb(p['total'])}  ·  " + " + ".join(bits)


def markdown(d: dict, title: str = "", ctx: int | None = None) -> str:
    """The estimate as Markdown for the TUI pager and `slmbench mem`."""
    from .i18n import T
    ram = system_memory()
    verdict = {"ok": T("✔ cabe holgado", "✔ fits comfortably"), "tight": T("⚠ justo", "⚠ tight"), "no": T("✗ no cabe", "✗ does not fit")}
    L = [f"# {T('Memoria estimada', 'Estimated memory')} · {title or d.get('model', '')}", ""]
    L.append(f"- **{T('Formato', 'Format')}**: {d.get('format', '?')}" + (f"  ·  **{T('Fichero', 'File')}**: `{d['file']}`" if d.get("file") else "")
             + (f"  ·  **{T('Memoria de este equipo', 'Memory of this machine')}**: {gb(ram)}" if ram else ""))
    if d.get("files"):
        L += ["", f"| {T('Fichero', 'File')} | {T('Pesos', 'Weights')} | KV cache | Overhead | Total | |", "|---|---:|---:|---:|---:|---|"]
        for f in d["files"]:
            p = parts(f)
            L.append(f"| `{f.get('file')}` | {gb(p['weights'])} | {gb(p['kv'])} | {gb(p['overhead'])} | **{gb(p['total'])}** | {verdict.get(fit(p['total'], ram), '')} |")
        asked, ctx = ctx, parts(d["files"][0])["ctx"] if d["files"] else None
        if ctx:
            L += ["", T(f"KV cache para {ctx:,} tokens de contexto", f"KV cache for {ctx:,} tokens of context")
                  + (T(" (el máximo del modelo).", " (the model's maximum).") if not asked else ".")]
    else:
        p = parts(d)
        L += ["", f"| {T('Parte', 'Part')} | {T('Memoria', 'Memory')} | |", "|---|---:|---|"]
        L.append(f"| {T('Pesos', 'Weights')} | {gb(p['weights'])} | " + (f"{p['params'] / 1e9:.2f}B params" if p["params"] else "") + " |")
        if p["kv"] is not None:
            L.append(f"| KV cache | {gb(p['kv'])} | {p['ctx']:,} ctx × {p['batch'] or 1} · {p['kv_dtype'] or ''} |")
        if p["overhead"] is not None:
            L.append(f"| Overhead | {gb(p['overhead'])} | {T('activaciones, grafos, runtime', 'activations, graphs, runtime')} |")
        L.append(f"| **Total** | **{gb(p['total'])}** | {verdict.get(fit(p['total'], ram), '')} |")
    for w in d.get("warnings") or []:
        L += ["", f"> ⚠ {w}"]
    L += ["", T("Estimación de hf-mem (github.com/alvarobartt/hf-mem) a partir de los metadatos, sin descargar los pesos.",
                "Estimate by hf-mem (github.com/alvarobartt/hf-mem) from the metadata, without downloading the weights.")]
    return "\n".join(L)
