"""Configuración persistente (JSON) compartida por el CLI y la TUI."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(os.environ.get("SLMBENCH_HOME") or Path(base) / "slmbench")


def config_path() -> Path:
    return Path(os.environ.get("SLMBENCH_CONFIG") or config_dir() / "config.json")


def _default_out() -> str:
    root = Path(__file__).resolve().parents[2]
    if (root / "bench" / "slmbench").is_dir():          # checkout del repo (uv sync / instalación editable)
        return str(root / "bench" / "results")
    return str(config_dir() / "results")                 # instalado como herramienta (uv tool install)


DEFAULT_OUT = _default_out()


@dataclass
class Settings:
    ui_lang: str = "en"                  # idioma de la TUI: en | es
    # --- destino del bench ---
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = ""                      # vacío = el primero de /v1/models
    api_key: str = ""
    # --- ejecución (todo pasa por lighteval) ---
    mode: str = "general"                # general | code | all (ver benchmarks/catalog.py)
    benchmarks: list = field(default_factory=list)   # claves elegidas; vacío = todas las del modo
    ld_extra_tasks: str = ""             # otras tareas de lighteval, separadas por comas (p. ej. arc:easy|0)
    ld_max_samples: int = 20             # muestras por benchmark (0 = dataset completo)
    ld_max_new_tokens: int = 2048        # máx. de tokens por respuesta (razonamiento incluido)
    reasoning_budget: int = 1024         # tokens de razonamiento por respuesta: -1 = sin límite, 0 = sin razonar, N = tope (thinking_budget_tokens).
                                         # Con 2048 de máximo deja ≈1024 para la respuesta; los modelos que no razonan lo ignoran
    temperature: float = 0.0
    passk_n: int = 5                     # benchmarks de programación: respuestas por problema para pass@1 y pass@5 (1 = solo pass@1, voraz)
    passk_temperature: float = 0.6       # temperatura de esas respuestas (con n > 1 hay que muestrear; con n = 1 se usa `temperature`)
    workers: int = 1                     # peticiones simultáneas (≤ slots -np del servidor)
    timeout: float = 600
    max_context: int = 32768
    extra_body: str = ""                 # JSON extra para cada petición
    label: str = ""
    label_model: str = ""             # model the label was set for: a label never carries over to another model ('*' = any)
    out_dir: str = DEFAULT_OUT
    # --- leaderboard ---
    leaderboard_auto: bool = True        # añadir cada ejecución al leaderboard
    leaderboard_dir: str = ""            # vacío = raíz del repo
    # --- servidor local (llama-server) ---
    # chat with the model (llama cli): last choices
    chat_model: str = ""                 # .gguf path or org/repo[:quant]; empty = the local model being evaluated
    chat_system: str = ""                # optional system prompt
    chat_ctx: int = 8192
    chat_reasoning_budget: int = 1024    # -1 = unlimited
    model_dirs: list = field(default_factory=list)   # carpetas adicionales donde buscar .gguf
    srv_source: str = "file"             # file | hf
    srv_model: str = ""                  # ruta del .gguf
    srv_hf: str = ""                     # usuario/repo[:quant]
    srv_alias: str = ""
    srv_port: int = 8080
    srv_ctx: int = 32768
    srv_ngl: str = "all"
    srv_fa: str = "on"
    srv_jinja: bool = True
    srv_parallel: int = 1
    srv_reasoning: str = "default"       # default | on | off
    srv_reasoning_budget: int = -1       # --reasoning-budget del servidor (-1 = sin límite)
    srv_extra: str = ""                  # argumentos adicionales (texto libre)

    def validate_extra(self):
        if self.extra_body.strip():
            v = json.loads(self.extra_body)
            if not isinstance(v, dict):
                raise ValueError("extra_body debe ser un objeto JSON")


def load() -> Settings:
    s = Settings()
    p = config_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
        known = {f.name for f in fields(Settings)}
        for k, v in data.items():
            if k in known:
                setattr(s, k, v)
    for k in ("out_dir", "leaderboard_dir"):        # "~/results" in the config file means the home folder, not a "~" folder
        if isinstance(getattr(s, k), str) and getattr(s, k).startswith("~"):
            setattr(s, k, os.path.expanduser(getattr(s, k)))
    return s


def save(s: Settings) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(s), indent=1, ensure_ascii=False))
    try:
        os.chmod(p, 0o600)   # puede contener una clave de API
    except OSError:
        pass
    return p
