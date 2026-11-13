"""Art for the TUI's home screen: the "slmbench" wordmark (lowercase letters 7 pixels tall, drawn with half blocks, at 1× or
2×), with "slm" in the logo's yellow -> orange -> pink gradient, and a gradient "benchmark bars" strip under it.
Only block characters are used, which every terminal draws with the same width.
"""
from __future__ import annotations

TAGLINE = "Where small models prove their worth."


# --- logotipo: letras en minúscula de 7 píxeles de alto, pintadas con medios bloques (4 filas) ----
_GLYPHS = {
    "s": ["....", "....", ".###", "#...", ".##.", "...#", "###."],
    "l": ["#.", "#.", "#.", "#.", "#.", "#.", ".#"],
    "m": [".....", ".....", "####.", "#.#.#", "#.#.#", "#.#.#", "#.#.#"],
    "b": ["#...", "#...", "###.", "#..#", "#..#", "#..#", "###."],
    "e": ["....", "....", ".##.", "#..#", "####", "#...", ".###"],
    "n": ["....", "....", "###.", "#..#", "#..#", "#..#", "#..#"],
    "c": ["....", "....", ".###", "#...", "#...", "#...", ".###"],
    "h": ["#...", "#...", "###.", "#..#", "#..#", "#..#", "#..#"],
}
# color de cada letra: «slm» en degradado amarillo → naranja → rosa (como el logo), «bench» en negrita
WORD_COLORS = {"s": "z", "l": "n", "m": "q"}


def _scaled(ch: str, scale: int) -> list[str]:
    g = _GLYPHS[ch]
    rows = ["".join(c * scale for c in row) for row in g]
    return [r for r in rows for _ in range(scale)]


def wordmark(word: str = "slmbench", scale: int = 1) -> list[list[tuple[str, str]]]:
    """Rows of (text, colour key) segments of the wordmark; scale 2 doubles every pixel (twice as wide and tall)."""
    glyphs = [_scaled(ch, scale) for ch in word]
    height = len(glyphs[0])
    height += height % 2
    rows = []
    for r in range(0, height, 2):
        segs = []
        for i, (ch, g) in enumerate(zip(word, glyphs)):
            blank = "." * len(g[0])
            top = g[r] if r < len(g) else blank
            bot = g[r + 1] if r + 1 < len(g) else blank
            cell = "".join("█" if t == "#" and b == "#" else "▀" if t == "#" else "▄" if b == "#" else " " for t, b in zip(top, bot))
            segs.append((cell, WORD_COLORS.get(ch, "B")))
            if i < len(word) - 1:
                segs.append((" " * scale, "B"))
        rows.append(segs)
    return rows


def wordmark_width(word: str = "slmbench", scale: int = 1) -> int:
    return (sum(len(_GLYPHS[c][0]) for c in word) + len(word) - 1) * scale


BAR_GRADIENT = [226, 220, 214, 208, 209, 203, 204, 205, 199, 170, 135, 99, 69, 75]   # yellow -> orange -> pink -> violet -> blue
_LEVELS = "▁▂▃▄▅▆▇█"


def bars(width: int) -> list[tuple[str, int]]:
    """A gradient "benchmark results" strip: (character, 256-colour) per column, rising and falling like a bar chart."""
    import math
    out = []
    for x in range(width):
        h = 0.5 + 0.32 * math.sin(x * 0.55) + 0.18 * math.sin(x * 1.7 + 1.0)
        level = _LEVELS[max(0, min(len(_LEVELS) - 1, int(h * len(_LEVELS))))]
        out.append((level, BAR_GRADIENT[int(x * (len(BAR_GRADIENT) - 1) / max(1, width - 1))]))
    return out


# colores del logotipo: clave -> (color de 256, negrita)
PALETTE = {
    "z": (220, True),    # «s» amarillo
    "n": (209, True),    # «l» naranja
    "q": (205, True),    # «m» rosa
    "B": (None, True),   # «bench»: color del terminal, negrita
}


# frases de la pantalla de inicio (una al azar en cada arranque): (inglés, español)
QUIPS = [
    ("Small model, big opinions.", "Modelo pequeño, opiniones grandes."),
    ("Fewer parameters, more personality.", "Menos parámetros, más personalidad."),
    ("Quantized, not compromised.", "Cuantizado, no acomplejado."),
    ("Q4_K_M: because RAM doesn't grow on trees.", "Q4_K_M: porque la RAM no crece en los árboles."),
    ("No GPU clusters were harmed in this evaluation.", "Ningún clúster de GPUs sufrió daños en esta evaluación."),
    ("It's not hallucinating. It's being creative.", "No está alucinando. Está siendo creativo."),
    ("Temperature 0. Nerves of steel.", "Temperatura 0. Nervios de acero."),
    ("Thinking… thinking… still thinking… (try --reasoning-budget)", "Pensando… pensando… sigue pensando… (prueba --reasoning-budget)"),
    ("Fits in your laptop. Fits in your heart.", "Cabe en tu portátil. Cabe en tu corazón."),
    ("Every token counts. Literally: we count them.", "Cada token cuenta. Literalmente: los contamos."),
    ("pass@1 or it didn't happen.", "pass@1 o no ha pasado."),
    ("Trust, but verify. Then verify 800 more times.", "Confía, pero verifica. Y luego verifica 800 veces más."),
    ("Zero cloud bills were issued today.", "Hoy no ha llegado ninguna factura de la nube."),
    ("Size isn't everything. Benchmarks are.", "El tamaño no lo es todo. Los benchmarks sí."),
    ("Small enough to run locally, brave enough to be benchmarked.", "Lo bastante pequeño para tu máquina, lo bastante valiente para un benchmark."),
]
