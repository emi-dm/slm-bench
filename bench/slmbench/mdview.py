"""Renderizado del Markdown de slmbench (informes, comparaciones, leaderboard, fichas) para la TUI.

`render(md, width)` devuelve una lista de líneas; cada línea es una lista de segmentos `(texto, estilo)`. La TUI traduce los estilos
a colores de curses. Cubre lo que generan los informes: títulos, tablas (alineadas, con bordes y porcentajes coloreados), listas,
citas, bloques de código, negritas y HTML sencillo (<sub>, <br>, <details>…).
"""
from __future__ import annotations

import re
import textwrap
import unicodedata

EMOJI_DOTS = "🟢🟡🟠🔴⚪"
PCT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s?%")


def dwidth(text: str) -> int:
    """Ancho en columnas de terminal (los caracteres anchos ocupan 2, los combinantes 0)."""
    w = 0
    for ch in text:
        if unicodedata.combining(ch) or ch in "\u200d\ufe0f":
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def pad(text: str, width: int, align: str = "l") -> str:
    gap = max(0, width - dwidth(text))
    if align == "r":
        return " " * gap + text
    if align == "c":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def clean_inline(text: str) -> str:
    """Quita marcas Markdown/HTML de una celda o línea (las negritas se tratan aparte)."""
    # sin selector de variación: «⚠️» (emoji) ocupa 2 columnas en muchos terminales y descuadra las tablas; «⚠» ocupa 1
    t = text.replace("\ufe0f", "").replace("<br>", " · ").replace("<br/>", " · ")
    t = re.sub(r"<sub>(.*?)</sub>", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)            # imágenes
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)        # enlaces -> texto
    t = t.replace("`", "")
    for d in EMOJI_DOTS:
        t = t.replace(d + " ", "").replace(d, "")
    return t.strip()


def inline_segments(text: str, base: str = "normal") -> list[tuple[str, str]]:
    """Separa **negritas** y `código` en segmentos con estilo."""
    t = clean_inline(text.replace("`", "\x00")).replace("\x00", "`")
    segs: list[tuple[str, str]] = []
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", t):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            segs.append((part[2:-2], "bold"))
        elif part.startswith("`") and part.endswith("`"):
            segs.append((part[1:-1], "code"))
        else:
            segs.append((part, base))
    return segs


def score_style(cell: str) -> str | None:
    m = PCT_RE.search(cell)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return "good" if v >= 85 else "mid" if v >= 65 else "warn" if v >= 40 else "bad"


def split_row(line: str) -> list[str]:
    """Divide una fila de tabla por «|» sin romper el contenido entre `comillas invertidas`."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, cur, code = [], "", False
    for ch in line:
        if ch == "`":
            code = not code
        if ch == "|" and not code:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
    cells.append(cur)
    return [c.strip() for c in cells]


def render_table(rows: list[str]) -> list[list[tuple[str, str]]]:
    header = split_row(rows[0])
    aligns = ["l"] * len(header)
    body_lines = rows[1:]
    if body_lines and re.fullmatch(r"\|?[\s:\-|]+\|?", body_lines[0].strip()):
        for i, spec in enumerate(split_row(body_lines[0])[: len(header)]):
            spec = spec.strip()
            aligns[i] = "r" if spec.endswith(":") and not spec.startswith(":") else "c" if spec.startswith(":") and spec.endswith(":") else "l"
        body_lines = body_lines[1:]
    body = [split_row(r) for r in body_lines]
    ncol = max([len(header)] + [len(r) for r in body])
    header += [""] * (ncol - len(header))
    aligns += ["l"] * (ncol - len(aligns))
    raw = [[clean_inline(c) for c in header]] + [[clean_inline(c) for c in r] + [""] * (ncol - len(r)) for r in body]
    bold = [[False] * ncol] + [["**" in c for c in r] + [False] * (ncol - len(r)) for r in body]
    widths = [max(dwidth(r[i].replace("**", "")) for r in raw) for i in range(ncol)]
    out = []
    border = "─"
    out.append([("┌" + "┬".join(border * (w + 2) for w in widths) + "┐", "border")])
    for ri, r in enumerate(raw):
        line: list[tuple[str, str]] = [("│", "border")]
        for i, cell in enumerate(r):
            text = cell.replace("**", "")
            style = "thead" if ri == 0 else (score_style(text) or ("dim" if text in ("–", "-", "") else "bold" if bold[ri][i] else "normal"))
            line.append((" " + pad(text, widths[i], aligns[i] if ri else "l") + " ", style))
            line.append(("│", "border"))
        out.append(line)
        if ri == 0:
            out.append([("├" + "┼".join(border * (w + 2) for w in widths) + "┤", "border")])
    out.append([("└" + "┴".join(border * (w + 2) for w in widths) + "┘", "border")])
    return out


def render(md: str, width: int = 100) -> list[list[tuple[str, str]]]:
    lines = md.split("\n")
    out: list[list[tuple[str, str]]] = []
    i = 0
    in_code = False
    fence = ""

    def wrap_add(text: str, style: str = "normal", indent: str = "", first: str | None = None):
        segs_text = clean_inline(text) if "**" not in text and "`" not in text else None
        if segs_text is not None and dwidth(segs_text) + len(first or indent) > width:
            for j, chunk in enumerate(textwrap.wrap(segs_text, max(20, width - len(indent))) or [""]):
                out.append([((first if (j == 0 and first is not None) else indent) + chunk, style)])
            return
        prefix = first if first is not None else indent
        segs = inline_segments(text, style)
        total = sum(dwidth(t) for t, _ in segs)
        if total + len(prefix) <= width:
            out.append([(prefix, style)] + segs)
        else:                                                  # línea larga con formato: se parte como texto plano
            plain = "".join(t for t, _ in segs)
            for j, chunk in enumerate(textwrap.wrap(plain, max(20, width - len(indent))) or [""]):
                out.append([((prefix if j == 0 else indent) + chunk, style)])

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if in_code:
            if s.startswith(fence):
                in_code = False
                out.append([])
            else:
                room = max(20, width - 6)
                chunks, rest = [], line
                while len(rest) > room:                     # wrap at a space when there is one near the end, else hard
                    cut = rest.rfind(" ", 0, room + 1)
                    cut = cut + 1 if cut >= room * 0.6 else room
                    chunks.append(rest[:cut].rstrip() if cut != room else rest[:cut])
                    rest = rest[cut:]
                chunks.append(rest)
                for chunk in chunks:
                    out.append([("  │ ", "border"), (chunk, "code")])
            i += 1
            continue
        m = re.match(r"^(`{3,})", s)
        if m:
            in_code, fence = True, m.group(1)
            i += 1
            continue
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.extend(render_table(block))
            out.append([])
            continue
        i += 1
        if re.match(r"^<(a id|p align|/?details|img)\b", s) and not re.search(r"<summary>", s):
            txt = clean_inline(s)
            if txt and not s.startswith(("<a id", "<img")):
                out.append([(txt, "h3")])
            continue
        if s.startswith("<summary>") or "<summary>" in s:
            out.append([("▸ " + clean_inline(s), "h3")])
            continue
        if s.startswith("<h1"):
            out.append([(clean_inline(s), "h1")])
            out.append([("═" * min(width, max(20, dwidth(clean_inline(s)))), "h1")])
            continue
        if s in ("---", "***"):
            out.append([("─" * min(width, 60), "border")])
            continue
        hm = re.match(r"^(#{1,4})\s+(.*)", s)
        if hm:
            level, text = len(hm.group(1)), clean_inline(hm.group(2)).replace("**", "")
            if out and out[-1]:
                out.append([])
            if level == 1:
                out.append([(text, "h1")])
                out.append([("═" * min(width, max(20, dwidth(text))), "h1")])
            elif level == 2:
                out.append([(text, "h2")])
                out.append([("─" * min(width, max(10, dwidth(text))), "h2")])
            else:
                out.append([(text, "h3")])
            continue
        if s.startswith(">"):
            wrap_add(s.lstrip("> ").strip(), "quote", indent="  ┃ ", first="  ┃ ")
            continue
        bm = re.match(r"^(\s*)[-*•]\s+(.*)", line)
        if bm:
            ind = " " * len(bm.group(1))
            wrap_add(bm.group(2), "normal", indent=ind + "    ", first=ind + "  • ")
            continue
        if not s:
            if out and out[-1]:
                out.append([])
            continue
        lead = line[: len(line) - len(line.lstrip())]
        wrap_add(s, "normal", indent=lead, first=lead)
    while out and not out[-1]:
        out.pop()
    return out
