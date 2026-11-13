"""Idioma de la interfaz (TUI y fichas de las pruebas): inglés por defecto, español opcional.

El texto fuente está en español; las traducciones al inglés viven en `i18n_en.py` (datos). `tr()` traduce una cadena exacta o, si
lleva partes variables, sustituye los fragmentos conocidos. Los informes salen en el idioma de la interfaz; el leaderboard, en inglés.
"""
from __future__ import annotations

LANGS = {"en": "English", "es": "Español"}
_lang = "en"


def set_lang(lang: str):
    global _lang
    _lang = lang if lang in LANGS else "en"


def get_lang() -> str:
    return _lang


def T(es: str, en: str) -> str:
    """Texto en el idioma activo (para textos nuevos, sin pasar por el diccionario)."""
    return en if _lang == "en" else es


def _data():
    try:
        from . import i18n_en
        return i18n_en
    except Exception:          # sin traducciones: se queda en español
        return None


# Textos añadidos después de i18n_en.py (o que no encajan en él)
EXTRA_UI = {
    "MODELO A EVALUAR": "MODEL TO EVALUATE",
    "Presupuesto de razonamiento": "Reasoning budget",
    "Presupuesto de razonamiento (--reasoning-budget)": "Reasoning budget (--reasoning-budget)",
    "-1 = sin límite · 0 = sin razonar · N = tope de tokens de razonamiento por respuesta (llama-server: thinking_budget_tokens)":
        "-1 = unlimited · 0 = no reasoning · N = max reasoning tokens per answer (llama-server: thinking_budget_tokens)",
    "Tope de tokens de razonamiento para todo el servidor: -1 = sin límite, 0 = sin razonar.":
        "Reasoning token cap for the whole server: -1 = unlimited, 0 = no reasoning.",
    "Interfaz": "Interface", "Idioma de la interfaz": "Interface language",
    "También con la tecla L en el menú principal.": "Also with the L key in the main menu.",
    "Informe": "Report", "Prueba": "Test",
    "[s] Sí    [n] No": "[y] Yes    [n] No", "   (Enter = Sí)": "   (Enter = Yes)",
    "modelos pequeños · EN + ES · basado en lighteval": "small models · EN + ES · built on lighteval",
    "IFEval, LiveCodeBench, GSM8K, MATH-500 con lighteval de Hugging Face": "IFEval, LiveCodeBench, GSM8K, MATH-500 with Hugging Face's lighteval",
    "↑↓ mover · Enter o 1-7 elegir · L idioma (EN/ES) · r reconectar · q salir": "↑↓ move · Enter or 1-7 select · L language (EN/ES) · r reconnect · q quit",
    "○ lighteval no está instalado: ejecuta `uv sync` en el repo (o pulsa «Instalar lighteval»)":
        '○ lighteval is not installed: run `uv sync` in the repo (or press "Install lighteval")',
    "Instalar lighteval": "Install lighteval",
    "(vacío)": "(empty)",
}
EXTRA_PHRASES = [
    ("↑↓ PgUp PgDn desplazar · ←→ horizontal · g/G inicio/fin · q salir", "↑↓ PgUp PgDn scroll · ←→ horizontal · g/G start/end · q quit"),
    ("  ·  fila en el leaderboard: ", "  ·  leaderboard row: "),
    (" (incluido con uv sync)", " (included with uv sync)"),
    ("servidor ", "server "),
]


_sorted = None


def _phrases(d):
    """Fragmentos de ambas fuentes, del más largo al más corto (los genéricos no deben pisar a los específicos)."""
    global _sorted
    if _sorted is None:
        _sorted = sorted(EXTRA_PHRASES + list(d.PHRASES), key=lambda p: -len(p[0]))
    return _sorted


def tr(text):
    """Traduce texto de la interfaz al idioma activo (no toca datos del usuario)."""
    if _lang == "es" or not isinstance(text, str) or not text:
        return text
    hit = EXTRA_UI.get(text)
    if hit is not None:
        return hit
    d = _data()
    if d is None:
        return text
    hit = d.UI.get(text)
    if hit is not None:
        return hit
    out = text
    for es, en in _phrases(d):
        if es in out:
            out = out.replace(es, en)
    return out.replace("«", '"').replace("»", '"')


def about(name: str, mod) -> dict:
    base = getattr(mod, "ABOUT", {}) if mod is not None else {}
    if _lang == "es":
        return base
    d = _data()
    return {**base, **(getattr(d, "ABOUT", {}).get(name, {}) if d else {})}


def description(name: str, default: str = "") -> str:
    if _lang == "es":
        return default
    d = _data()
    return (getattr(d, "DESCRIPTIONS", {}).get(name) if d else None) or default


def expected(task) -> str:
    if _lang == "es":
        return task.expected
    d = _data()
    return (getattr(d, "EXPECTED", {}).get(task.id.split("@")[0]) if d else None) or task.expected


def lighteval_card(default: dict) -> dict:
    if _lang == "es":
        return default
    d = _data()
    return {**default, **(getattr(d, "LIGHTEVAL_CARD", {}) if d else {})}


def lighteval_curated(key: str, default: list) -> list:
    if _lang == "es":
        return default
    d = _data()
    return (getattr(d, "LIGHTEVAL_CURATED", {}).get(key) if d else None) or default
