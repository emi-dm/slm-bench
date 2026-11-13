"""Descubrimiento de modelos GGUF y gestión de un `llama-server` local."""
from __future__ import annotations

import json
import glob
import platform
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .config import Settings, config_dir


class _NoCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """urllib keeps every header on a redirect, Authorization included, even to another host. Drop it then: the API key
    goes only to the server it was configured for."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoCrossHostAuth)

DEFAULT_ROOTS = ["~/.cache/huggingface/hub", "~/.cache/llama.cpp", "~/.lmstudio/models", "~/models", "~/.local/share/llama.cpp"]
SKIP_PREFIXES = ("mmproj", "ggml-vocab")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _is_wsl() -> bool:
    try:
        return "microsoft" in platform.release().lower() or os.path.isdir("/mnt/c/Windows")
    except OSError:
        return False


def model_roots(extra_dirs: list[str] | None = None) -> list[Path]:
    """Folders searched for .gguf files: the usual caches, the ones set through environment variables, macOS's llama.cpp
    cache and, on WSL, the usual Windows locations (LM Studio, HF cache, llama.cpp) of each Windows user."""
    roots = list(DEFAULT_ROOTS) + ["~/Library/Caches/llama.cpp"]
    env = os.environ
    for var, sub in (("HF_HUB_CACHE", ""), ("HUGGINGFACE_HUB_CACHE", ""), ("HF_HOME", "hub"), ("LLAMA_CACHE", ""),
                     ("XDG_CACHE_HOME", "huggingface/hub"), ("XDG_CACHE_HOME", "llama.cpp")):
        if env.get(var):
            roots.append(os.path.join(env[var], sub) if sub else env[var])
    if _is_wsl():
        for user in sorted(glob.glob("/mnt/c/Users/*")):
            if os.path.basename(user) in ("Public", "Default", "Default User", "All Users"):
                continue
            roots += [f"{user}/.lmstudio/models", f"{user}/.cache/lm-studio/models", f"{user}/.cache/huggingface/hub",
                      f"{user}/.cache/llama.cpp", f"{user}/AppData/Local/llama.cpp", f"{user}/models"]
    out, seen = [], set()
    for r in roots + list(extra_dirs or []):
        p = Path(os.path.expanduser(r))
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def discover_models(extra_dirs: list[str] | None = None, max_depth: int = 7) -> list[dict]:
    """Busca ficheros .gguf (sin mmproj, vocabularios ni descargas incompletas)."""
    roots = model_roots(extra_dirs)
    seen, out = set(), []
    for root in roots:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dp, dns, fns in os.walk(root):
            if len(Path(dp).parts) - base_depth >= max_depth:
                dns[:] = []
            dns[:] = [d for d in dns if d not in ("blobs", ".git", "node_modules")]
            for fn in fns:
                if not fn.lower().endswith(".gguf") or fn.lower().startswith(SKIP_PREFIXES):
                    continue
                p = Path(dp) / fn
                try:
                    real = p.resolve()
                    size = real.stat().st_size
                except OSError:
                    continue
                if real in seen or size < 1_000_000:
                    continue
                seen.add(real)
                name = fn[:-5]
                # nombre legible: para la caché de HF usa «repo / fichero»
                parts = p.parts
                for part in parts:
                    if part.startswith("models--"):
                        name = f"{part[8:].replace('--', '/')} · {fn[:-5]}"
                        break
                out.append({"name": name, "path": str(p), "size": size})
    out.sort(key=lambda m: m["name"].lower())
    return out


def server_command() -> list[str] | None:
    """How to start llama.cpp's server: `llama-server`, or `llama serve` in builds that ship the unified `llama` command
    (same arguments). None if neither is on the PATH."""
    if shutil.which("llama-server"):
        return [shutil.which("llama-server")]
    llama = shutil.which("llama")
    if llama:
        # Unified CLI: `llama <command>`. Its help lists "serve"; some builds exit non-zero when printing help, so the exit
        # code is ignored. If the help can't be read, `llama serve` is still the best guess when there's no llama-server.
        try:
            r = subprocess.run([llama, "--help"], capture_output=True, timeout=20)
            text = (r.stdout + r.stderr).decode(errors="replace")
            if text and not re.search(r"^\s*serve\b", text, re.M) and re.search(r"^\s*(cli|version)\b", text, re.M):
                return None                            # a `llama` command with subcommands but no server
        except (OSError, subprocess.SubprocessError):
            pass
        return [llama, "serve"]
    return None


def _unified(sub: str) -> list[str] | None:
    """`llama <sub>` from the unified llama.cpp CLI, if installed (exit code of --help ignored, see server_command)."""
    llama = shutil.which("llama")
    if not llama:
        return None
    try:
        r = subprocess.run([llama, "--help"], capture_output=True, timeout=20)
        text = (r.stdout + r.stderr).decode(errors="replace")
        if text and not re.search(rf"^\s*{sub}\b", text, re.M) and re.search(r"^\s*(serve|cli|version)\b", text, re.M):
            return None
    except (OSError, subprocess.SubprocessError):
        pass
    return [llama, sub]


def cli_command() -> list[str] | None:
    """llama.cpp's interactive chat: `llama-cli`, or `llama cli` in builds with the unified `llama` command."""
    if shutil.which("llama-cli"):
        return [shutil.which("llama-cli")]
    return _unified("cli")


def chat_command(s: Settings, model: str, system: str = "", ctx: int = 8192, reasoning_budget: int = -1) -> list[str]:
    """Command for an interactive chat with a GGUF (a path, or an `org/repo[:quant]` from Hugging Face) via llama cli."""
    cmd = list(cli_command() or ["llama-cli"])
    cmd += ["-m", model] if os.path.isfile(model) else ["-hf", model]
    cmd += ["-c", str(ctx), "-ngl", str(s.srv_ngl), "-fa", s.srv_fa, "--jinja"]
    if reasoning_budget is not None and reasoning_budget >= 0:
        cmd += ["--reasoning-budget", str(reasoning_budget)]
    if system.strip():
        cmd += ["-sys", system]
    return cmd


def find_binary() -> str | None:
    """Human-readable form of server_command() (e.g. "llama serve"), or None."""
    cmd = server_command()
    return " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)) if cmd else None


INSTALL_HINT = ("install llama.cpp: `brew install llama.cpp` (macOS/Linux), a release from "
                "https://github.com/ggml-org/llama.cpp/releases, or build it; slmbench uses `llama-server` or `llama serve`")


def probe(base_url: str, api_key: str = "", timeout: float = 4) -> list[str]:
    """Devuelve los ids de /models; lanza excepción si no hay conexión."""
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    with _OPENER.open(req, timeout=timeout) as r:
        data = json.load(r)
    return [m["id"] for m in data.get("data", [])]


def model_meta(base_url: str, model_id: str, timeout: float = 2) -> dict:
    """Parámetros, tamaño, cuantización y versión de llama.cpp que expone llama-server (vacío con otros servidores)."""
    import re
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]

    def get(url):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            return {}
    out: dict = {}
    models = get(root + "/v1/models").get("data", [])
    m = next((x for x in models if x.get("id") == model_id), models[0] if models else {})
    meta = m.get("meta") or {}
    if meta.get("n_params"):
        out["params_b"] = round(meta["n_params"] / 1e9, 2)
    if meta.get("size"):
        out["size_gb"] = round(meta["size"] / 1024 ** 3, 2)
    if meta.get("ftype"):
        q = meta["ftype"].replace(" - ", "_").replace(" ", "_")
        for long_, short in (("_Medium", "_M"), ("_Small", "_S"), ("_Large", "_L"), ("_XLarge", "_XL")):
            q = q.replace(long_, short)
        out["quant"] = q
    if meta.get("n_ctx_train"):
        out["ctx_train"] = meta["n_ctx_train"]
    props = get(root + "/props")
    if props.get("build_info"):
        out["backend"] = f"llama.cpp {props['build_info']}"
    if props.get("total_slots"):
        out["slots"] = props["total_slots"]
    if not out.get("quant"):
        mq = re.search(r"(IQ\d_\w+|Q\d_[A-Z0-9_]+|BF16|F16|MXFP4)", model_id or "", re.I)
        if mq:
            out["quant"] = mq.group(1).upper()
    return out


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# ---------- proceso gestionado ----------

def _state_file() -> Path:
    return config_dir() / "server.json"


def clean_model_id(model_id: str) -> str:
    """A server started without --alias reports the GGUF's path as the model id. Turn it into a name: a Hugging Face cache
    path -> "org/model-GGUF:Q4_K_M"-like id (org/repo plus the file name for the quant), any other path -> the file name."""
    mid = model_id or ""
    if not (mid.startswith(("/", "~", "\\")) or mid.lower().endswith(".gguf") or ":\\" in mid[:3]):
        return mid
    parts = Path(mid.replace("\\", "/")).parts
    stem = Path(mid.replace("\\", "/")).stem
    repo = next((x[8:].replace("--", "/") for x in parts if x.startswith("models--")), "")
    return f"{repo}:{stem}" if repo else stem


def default_alias(s: Settings) -> str:
    """Id con el que se sirve el modelo (y fila del leaderboard junto con la quant).
    GGUF de la caché de Hugging Face -> «org/modelo» (sin -GGUF); -hf org/repo:quant -> «org/modelo»; otro fichero -> nombre del fichero."""
    import re
    if s.srv_source == "hf":
        repo = s.srv_hf.split(":")[0]
    else:
        repo = next((p[8:].replace("--", "/") for p in Path(s.srv_model).parts if p.startswith("models--")), "")
        if not repo:
            return Path(s.srv_model).stem
    return re.sub(r"(?i)[-_.]gguf$", "", repo)


def build_command(s: Settings, binary: str | list[str]) -> list[str]:
    cmd = list(binary) if isinstance(binary, (list, tuple)) else [binary]
    if s.srv_source == "hf":
        cmd += ["-hf", s.srv_hf]
    else:
        cmd += ["-m", s.srv_model]
    alias = s.srv_alias or default_alias(s)
    cmd += ["--alias", alias, "--host", "127.0.0.1", "--port", str(s.srv_port), "-c", str(s.srv_ctx),
            "-ngl", str(s.srv_ngl), "-fa", s.srv_fa, "-np", str(s.srv_parallel)]
    cmd.append("--jinja" if s.srv_jinja else "--no-jinja")
    if s.srv_reasoning in ("on", "off"):
        cmd += ["--reasoning", s.srv_reasoning]
    if s.srv_reasoning_budget is not None and s.srv_reasoning_budget >= 0:
        cmd += ["--reasoning-budget", str(s.srv_reasoning_budget)]
    if s.srv_extra.strip():
        cmd += shlex.split(s.srv_extra)
    return cmd


def alias_of(cmd: list[str]) -> str:
    return cmd[cmd.index("--alias") + 1] if "--alias" in cmd else ""


def managed_state() -> dict | None:
    """Estado del servidor lanzado por la TUI, si sigue vivo."""
    p = _state_file()
    if not p.exists():
        return None
    try:
        st = json.loads(p.read_text())
        os.kill(st["pid"], 0)
    except Exception:
        try:
            p.unlink()
        except OSError:
            pass
        return None
    return st


def start(s: Settings) -> dict:
    binary = server_command()
    if not binary:
        raise RuntimeError("llama.cpp's server (`llama-server` or `llama serve`) is not on the PATH: " + INSTALL_HINT)
    if s.srv_source == "file" and not (s.srv_model and os.path.isfile(s.srv_model)):
        raise RuntimeError("Elige un fichero .gguf válido.")
    if s.srv_source == "hf" and not s.srv_hf.strip():
        raise RuntimeError("Indica el repositorio de Hugging Face (usuario/repo[:quant]).")
    if managed_state():
        raise RuntimeError("Ya hay un servidor gestionado en marcha; páralo primero.")
    if health(s.srv_port):
        raise RuntimeError(f"El puerto {s.srv_port} ya está ocupado por otro servidor.")
    cmd = build_command(s, binary)
    logdir = config_dir() / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"llama-server-{s.srv_port}.log"
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    st = {"pid": proc.pid, "port": s.srv_port, "alias": alias_of(cmd), "cmd": cmd, "log": str(log), "started": time.time()}
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(st))
    return st


def stop() -> bool:
    st = managed_state()
    if not st:
        return False
    try:
        os.killpg(os.getpgid(st["pid"]), signal.SIGTERM)
    except Exception:
        try:
            os.kill(st["pid"], signal.SIGTERM)
        except Exception:
            pass
    for _ in range(30):
        time.sleep(0.2)
        try:
            os.kill(st["pid"], 0)
        except OSError:
            break
    else:
        try:
            os.kill(st["pid"], signal.SIGKILL)
        except OSError:
            pass
    try:
        _state_file().unlink()
    except OSError:
        pass
    return True


def tail(path: str, n: int = 200) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            data = f.read().decode(errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except OSError:
        return "(sin log)"
