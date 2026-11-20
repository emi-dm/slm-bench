"""TUI de slmbench (curses, solo librería estándar).

Pantallas: menú principal · modelo a evaluar (GGUF local o servidor) · lanzar benchmarks por modo (lighteval, progreso en vivo)
· benchmarks (fichas con tarea de muestra) · resultados y comparación · leaderboard · ajustes. Abajo se avisa de las ejecuciones
de slmbench en marcha (en este u otro proceso).
"""
from __future__ import annotations

import contextlib
import curses
import json
import locale
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from . import __version__, art, config, engine, i18n, leaderboard, mdview, servers
from .benchmarks import catalog as cat
from .i18n import T
from .i18n import tr as _

os.environ.setdefault("ESCDELAY", "25")

C_OK, C_BAD, C_WARN, C_ACC, C_HEAD, C_DIM, C_MAG = 1, 2, 3, 4, 5, 6, 7

KEY_ESC = "\x1b"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class Field:
    key: str
    label: str
    kind: str = "text"          # text | int | float | bool | choice | action | section | info
    value: object = None
    choices: list = dc_field(default_factory=list)   # [(valor, texto)]
    hint: str = ""
    secret: bool = False
    placeholder: str = ""       # shown, muted, instead of «(empty)» when there's no value


SELECTABLE = ("text", "int", "float", "bool", "choice", "action")


_HOME = str(Path.home())
ASCII_ONLY = False
_ASCII = str.maketrans({**{c: "-" for c in "─━═"}, **{c: "|" for c in "│┃║"}, **{c: "+" for c in "┌┐└┘├┤┬┴┼╔╗╚╝"},
                        **{c: "#" for c in "█▀▄▌▐"}, "░": ".", "·": "-", "•": "*", "…": ".", "●": "*", "○": "o", "▶": ">",
                        "▸": ">", "‹": "<", "›": ">", "←": "<", "→": ">", "↑": "^", "↓": "v", "»": ">", "≤": "<", "⚠": "!",
                        "✔": "v", "✓": "v", "✗": "x", "–": "-", "—": "-", "«": '"', "╭": "+", "╮": "+", "╰": "+", "╯": "+",
                        "◆": "*", "◉": "*", "❯": ">", "▍": "|", "▎": "|", "▏": "|", "╌": "-", "⋯": ".", "■": "#", "▬": "=", **{c: "*" for c in SPIN}})


def _run_name(r: dict) -> str:
    """Results list: the model that was measured (+ quant), and the run's label only when it was a custom one."""
    from .servers import clean_model_id
    name = clean_model_id(r.get("model") or "?")
    q = r.get("quant") or ""
    if q and q.lower() not in name.lower():
        name += f" {q}"
    if r.get("custom_label"):                   # only a label set on purpose for this model (old runs may carry a stale one)
        name += f" ({r['custom_label']})"
    return name


def _home_short(text: str) -> str:
    """Muestra las rutas de la carpeta personal como «~/…» (pantallas más limpias y sin el nombre de usuario)."""
    return text.replace(_HOME + "/", "~/") if _HOME and _HOME != "/" and _HOME in text else text


def _italic() -> int:
    return getattr(curses, "A_ITALIC", 0)


def _basic(c: int) -> int:
    """Color de 256 -> el más parecido de los 8 básicos (terminales sin 256 colores)."""
    if c >= 232:
        return curses.COLOR_WHITE if c >= 244 else curses.COLOR_BLACK
    if c < 16:
        return c % 8
    c -= 16
    r, g, b = c // 36, (c // 6) % 6, c % 6
    if max(r, g, b) - min(r, g, b) <= 1:
        return curses.COLOR_WHITE if max(r, g, b) >= 3 else curses.COLOR_BLACK
    return (1 if r >= 3 else 0) | (2 if g >= 3 else 0) | (4 if b >= 3 else 0)


class UI:
    def __init__(self, scr):
        self.scr = scr
        curses.curs_set(0)
        scr.keypad(True)
        scr.idlok(False)                        # no scroll / insert-line tricks: some terminals leave ghost text with them
        scr.idcok(False)
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        if curses.COLORS >= 256:                # the slmbench logo palette: yellow -> orange -> pink -> violet -> blue, navy ink
            for i, fg in ((C_OK, 78), (C_BAD, 203), (C_WARN, 220), (C_ACC, 209), (C_DIM, 105), (C_MAG, 205)):
                curses.init_pair(i, fg, bg)
            curses.init_pair(C_HEAD, 17, 209)
        else:
            for i, fg in ((C_OK, curses.COLOR_GREEN), (C_BAD, curses.COLOR_RED), (C_WARN, curses.COLOR_YELLOW),
                          (C_ACC, curses.COLOR_YELLOW), (C_DIM, curses.COLOR_BLUE), (C_MAG, curses.COLOR_MAGENTA)):
                curses.init_pair(i, fg, bg)
            curses.init_pair(C_HEAD, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        try:                                    # clic en el indicador «● en marcha» del pie para ver la ejecución
            curses.mousemask(curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED | curses.BUTTON1_RELEASED)
        except curses.error:
            pass
        self._pairs: dict = {}                  # (fg, bg) -> par de color de curses, para el arte (art.py)
        self._working_since = None
        if os.environ.get("SLMBENCH_TRACE"):
            threading.Thread(target=self._watchdog, daemon=True).start()
        self.on_status = None                   # callback que abre la vista en vivo (lo fija App)
        self._status_x = None
        self._in_status = False

    def _watchdog(self, limit: float = 0.4):
        """SLMBENCH_TRACE=1: when the screen doesn't get back to waiting for a key within `limit` seconds, write where the
        main thread is stuck to ~/.config/slmbench/logs/tui-trace.log (one stack per freeze, with its duration)."""
        import sys
        import traceback
        from .config import config_dir
        log = config_dir() / "logs" / "tui-trace.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        main_id = threading.main_thread().ident
        seen = None
        while True:
            time.sleep(0.1)
            t0 = self._working_since
            if t0 is None or time.time() - t0 < limit or seen == t0:
                if t0 is None and seen is not None:
                    seen = None
                continue
            seen = t0
            frame = sys._current_frames().get(main_id)
            stack = "".join(traceback.format_stack(frame)[-12:]) if frame else "?"
            with open(log, "a") as fh:
                fh.write(f"--- {time.strftime('%H:%M:%S')} main thread busy for > {limit} s ---\n{stack}\n")

    # ---------- primitivas ----------
    def size(self):
        return self.scr.getmaxyx()

    def put(self, y, x, text, attr=0, maxw=None):
        h, w = self.size()
        if y < 0 or y >= h or x >= w:
            return
        room = (w - x - (1 if y == h - 1 else 0)) if maxw is None else min(maxw, w - x)
        if room <= 0:
            return
        try:
            text = _home_short(str(text))
            if ASCII_ONLY:                              # no UTF-8 locale: ASCII stand-ins instead of blank cells
                text = text.translate(_ASCII).encode("ascii", "replace").decode().replace("?", " ") if not text.isascii() else text
            self.scr.addstr(y, x, text[:room], attr)
        except curses.error:
            pass

    def color(self, fg, bg=None, bold=False):
        """Atributo de curses para colores de 256 (fg/bg; None = color del terminal), con respaldo para 8 colores."""
        key = (fg, bg)
        if key not in self._pairs:
            n = len(self._pairs) + 16
            if n >= curses.COLOR_PAIRS:
                return curses.A_BOLD if bold else 0
            conv = (lambda c: -1 if c is None else c) if curses.COLORS >= 256 else (lambda c: -1 if c is None else _basic(c))
            try:
                curses.init_pair(n, conv(fg), conv(bg))
            except curses.error:
                return curses.A_BOLD if bold else 0
            self._pairs[key] = n
        return curses.color_pair(self._pairs[key]) | (curses.A_BOLD if bold else 0)

    def draw_wordmark(self, y, x, scale: int = 1) -> int:
        """Draws the "slmbench" wordmark; returns the number of rows used."""
        rows = art.wordmark(scale=scale)
        for i, segs in enumerate(rows):
            xx = x
            for text, key in segs:
                fg, bold = art.PALETTE[key]
                self.put(y + i, xx, text, self.color(fg, None, bold))
                xx += len(text)
        return len(rows)

    def draw_bars(self, y, x, width: int):
        """The gradient "benchmark bars" strip under the wordmark."""
        for j, (ch, fg) in enumerate(art.bars(width)):
            self.put(y, x + j, ch, self.color(fg))

    def hline(self, y, ch="-", attr=0):
        h, w = self.size()
        self.put(y, 0, ch * (w - 1), attr)

    # palette of the chrome (256 colours; color() falls back to the 8 basic ones)
    BAR_BG, BADGE_BG, INK, MUTED, RULE, SEL_BG = 236, 209, 17, 245, 238, 237

    def header(self, title, sub=""):
        """Title bar: an orange «slmbench» badge, the screen's name, language and version; the subtitle under it."""
        h, w = self.size()
        bar = self.color(252, self.BAR_BG)
        self.put(0, 0, " " * (w - 1), bar)
        badge = " slmbench "
        self.put(0, 0, badge, self.color(self.INK, self.BADGE_BG, True))
        self.put(0, len(badge), "▌", self.color(self.BADGE_BG, self.BAR_BG))
        self.put(0, len(badge) + 2, _(title), self.color(255, self.BAR_BG, True))
        right = f"{i18n.get_lang().upper()} · v{__version__} "
        if w - len(right) - 1 > len(badge) + len(_(title)) + 4:
            self.put(0, w - len(right) - 1, right, self.color(self.MUTED, self.BAR_BG))
        if sub:
            self.put(1, 1, _(sub), curses.color_pair(C_DIM))

    def keyhints(self, y, x, text, maxw):
        """«↑↓ move · Enter select»: the key of each hint in the accent colour, its meaning muted."""
        segs = text.split(" · ")
        for i, seg in enumerate(segs):
            if x - 1 >= maxw:
                return
            if i:
                self.put(y, x, " · ", self.color(self.RULE), maxw=max(0, maxw - x))
                x += 3
            key, _sp, rest = seg.partition(" ")
            if seg.startswith("[") or not rest:
                key, rest = "", seg
            if key:
                self.put(y, x, key, curses.color_pair(C_ACC) | curses.A_BOLD, maxw=max(0, maxw - x))
                x += len(key) + 1
            self.put(y, x, rest, self.color(self.MUTED), maxw=max(0, maxw - x))
            x += len(rest)

    def footer(self, text):
        h, w = self.size()
        self.put(h - 1, 0, " " * (w - 1))
        status = running_status()
        self._status_x = max(2, w - len(status) - 2) if status else None
        hint = _(text)
        if status and len(hint) > self._status_x - 3:          # el indicador manda: se recorta la ayuda
            hint = hint[: max(0, self._status_x - 4)] + "…"
        self.keyhints(h - 1, 1, hint, (self._status_x - 2) if status else w - 1)
        if status:
            self.put(h - 1, self._status_x, status, curses.color_pair(C_WARN) | curses.A_BOLD)

    def hintline(self, text):
        """The help of the selected row, just above the footer."""
        h, w = self.size()
        if text:
            self.put(h - 3, 1, "▎", curses.color_pair(C_ACC))
            self.put(h - 3, 3, _(text), self.color(250) | _italic())

    def toast(self, text, kind=C_WARN):
        h, w = self.size()
        self.put(h - 2, 0, " " * (w - 1))
        self.put(h - 2, 1, _(text), curses.color_pair(kind) | curses.A_BOLD)

    def section(self, y, x, label, width):
        """◆ Label ╌╌╌╌ : the section title used by forms and the live view."""
        self.put(y, x, "◆ ", curses.color_pair(C_ACC))
        self.put(y, x + 2, label, curses.color_pair(C_ACC) | curses.A_BOLD)
        self.put(y, x + 3 + len(label), "╌" * max(0, width - len(label) - 4), self.color(self.RULE))

    def progress(self, y, x, width, frac):
        """A progress bar painted with the logo's gradient (yellow → pink → blue) over a dim track."""
        n = int(round(width * max(0.0, min(1.0, frac))))
        grad = art.bars(width)
        for j in range(width):
            if j < n:
                self.put(y, x + j, "█", self.color(grad[j][1]))
            else:
                self.put(y, x + j, "░", self.color(self.RULE))

    def sel_bar(self, y, x, width):
        """Background of the selected row: a soft band with an accent edge."""
        self.put(y, x, " " * max(0, width), self.color(255, self.SEL_BG))
        self.put(y, x, "▌", self.color(self.BADGE_BG, self.SEL_BG))

    def box(self, y, x, hh, ww, title="", attr=None):
        """Rounded box; returns nothing. attr = the border's attribute."""
        attr = curses.color_pair(C_ACC) if attr is None else attr
        t = f" {title} " if title else ""
        self.put(y, x, "╭─" + t + "─" * max(0, ww - 3 - len(t)) + "╮", attr)
        if t:
            self.put(y, x + 2, t, attr | curses.A_BOLD)
        for i in range(1, hh - 1):
            self.put(y + i, x, "│", attr)
            self.put(y + i, x + ww - 1, "│", attr)
        self.put(y + hh - 1, x, "╰" + "─" * (ww - 2) + "╯", attr)

    def key(self, timeout=-1):
        self._working_since = None              # back to waiting for the user: not frozen
        self.scr.timeout(timeout)
        try:
            k = self.scr.get_wch()
        except curses.error:
            return None
        finally:
            self._working_since = time.time()
        if k == curses.KEY_MOUSE:
            try:
                _id, x, y, _z, _b = curses.getmouse()
            except curses.error:
                return curses.KEY_RESIZE
            if y == self.size()[0] - 1 and self._status_x is not None and x >= self._status_x - 1:
                self.open_status()
            return curses.KEY_RESIZE            # para quien llama, un evento de ratón es solo «redibuja»
        if k == "\x12":                          # Ctrl+R: vista en vivo de la ejecución en marcha
            self.open_status()
            return curses.KEY_RESIZE
        return k

    def open_status(self):
        if self.on_status and not self._in_status and engine.active_runs():
            self._in_status = True
            try:
                self.on_status()
            finally:
                self._in_status = False

    # ---------- diálogos ----------
    def _dialog(self, title, lines, kind, buttons):
        """A centered box over the current screen: title, lines (the first one in the dialog's colour) and buttons."""
        h, w = self.size()
        ww = min(w - 4, max(44, max((len(l) for l in lines), default=0) + 6, len(buttons) + 6, len(_(title)) + 10))
        body = lines[: max(1, h - 8)]
        hh = len(body) + 5
        y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
        fill = self.color(252, 235)
        for i in range(hh):
            self.put(y + i, x, " " * ww, fill)
            if i:                                   # shadow
                self.put(y + i, x + ww, " ", self.color(None, 233))
        self.put(y + hh, x + 1, " " * ww, self.color(None, 233))
        border = self.color({C_BAD: 203, C_OK: 78, C_WARN: 220}.get(kind, 209), 235)
        self.box(y, x, hh, ww, _(title), border | curses.A_BOLD)
        for i, l in enumerate(body):
            self.put(y + 2 + i, x + 3, l[: ww - 5], (self.color({C_BAD: 203, C_OK: 78, C_WARN: 220}.get(kind, 209), 235, True)
                                                    if i == 0 else fill))
        self.put(y + hh - 2, x + ww - len(buttons) - 3, buttons, self.color(self.MUTED, 235))

    def choose(self, title, items, sel=0, footer=""):
        """A pop-up list over the current screen: items [(label, hint)]; ↑↓ move, Enter choose, q/Esc cancel.
        Returns the index chosen or None. The hint of the highlighted item is shown under the list."""
        footer = footer or T("↑↓ mover · Enter elegir · Esc cancelar", "↑↓ move · Enter choose · Esc cancel")
        while True:
            h, w = self.size()
            ww = min(w - 4, max(48, max((len(_(l)) for l, _h in items), default=0) + 10, len(footer) + 6, len(_(title)) + 10))
            hint = self._wrap(_(items[sel][1]), ww - 6)[:4] if items[sel][1] else []
            hh = len(items) + len(hint) + 6
            y, x = max(1, (h - hh) // 2), max(1, (w - ww) // 2)
            fill = self.color(252, 235)
            for i in range(hh):
                self.put(y + i, x, " " * ww, fill)
                if i:
                    self.put(y + i, x + ww, " ", self.color(None, 233))
            self.put(y + hh, x + 1, " " * ww, self.color(None, 233))
            self.box(y, x, hh, ww, _(title), self.color(209, 235) | curses.A_BOLD)
            for i, (label, _h) in enumerate(items):
                cur = i == sel
                self.put(y + 2 + i, x + 2, (" ▸ " if cur else "   ") + _(label).ljust(ww - 7)[: ww - 7],
                         self.color(255, self.SEL_BG, True) if cur else fill)
            for j, l in enumerate(hint):
                self.put(y + 3 + len(items) + j, x + 4, l, self.color(self.MUTED, 235))
            self.put(y + hh - 2, x + ww - len(footer) - 3, footer, self.color(self.MUTED, 235))
            self.scr.refresh()
            k = self.key()
            if k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(items)
            elif k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(items)
            elif k in ("\n", "\r", curses.KEY_ENTER):
                return sel
            elif k in ("q", "Q", KEY_ESC):
                return None
            elif k == curses.KEY_RESIZE:
                self.scr.erase()

    @staticmethod
    def _wrap(text, width=70):
        import textwrap
        out = []
        for part in str(text).split("\n"):
            out += textwrap.wrap(part, width, break_long_words=True, break_on_hyphens=False) or [""]
        return out

    def message(self, text, kind=C_ACC, title="Aviso"):
        lines = self._wrap(_(text), min(70, max(30, self.size()[1] - 12)))
        if self.size()[0] - 8 < len(lines):            # too long for a box: a pager
            self.pager(title, "\n".join(lines))
            return
        while True:
            self._dialog(title, lines, kind, _("Enter: continuar") if i18n.get_lang() == "es" else "Enter: continue")
            self.scr.refresh()
            k = self.key()
            if k == curses.KEY_RESIZE:
                self.scr.erase()
                continue
            if k is not None:
                return

    def confirm(self, text, default=False):
        lines = self._wrap(_(text), min(70, max(30, self.size()[1] - 12)))
        buttons = (T("[S] Sí", "[Y] Yes") + "   " + T("[n] No", "[n] No")) if default else (T("[s] Sí", "[y] Yes") + "   " + T("[N] No", "[N] No"))
        while True:
            self._dialog(T("Confirmar", "Confirm"), lines, C_WARN, buttons + "   " + T("Enter = ", "Enter = ") + (T("Sí", "Yes") if default else "No"))
            self.scr.refresh()
            k = self.key()
            if k == curses.KEY_RESIZE:
                self.scr.erase()
                continue
            if k in ("s", "S", "y", "Y"):
                return True
            if k in ("n", "N", KEY_ESC):
                return False
            if k in ("\n", "\r", curses.KEY_ENTER):
                return default

    def edit(self, prompt, initial="", secret=False, hint=""):
        """Editor de línea en la parte inferior. Devuelve el texto o None si se cancela con Esc."""
        buf, pos = list(initial), len(initial)
        curses.curs_set(1)
        try:
            while True:
                h, w = self.size()
                self.put(h - 3, 0, " " * (w - 1))
                self.hintline(hint or "Enter: aceptar · Esc: cancelar · Ctrl-U: borrar")
                self.put(h - 2, 0, " " * (w - 1))
                label = "❯ " + _(prompt) + " "
                shown = ("•" * len(buf)) if secret else "".join(buf)
                room = max(10, w - len(label) - 4)
                start = max(0, pos - room + 1)
                self.put(h - 2, 1, label, curses.color_pair(C_ACC) | curses.A_BOLD)
                self.put(h - 2, 1 + len(label), " " * (room + 1), self.color(255, self.BAR_BG))
                self.put(h - 2, 1 + len(label), shown[start:start + room], self.color(255, self.BAR_BG))
                try:
                    self.scr.move(h - 2, 1 + len(label) + pos - start)
                except curses.error:
                    pass
                self.scr.refresh()
                ch = self.key()
                if ch is None or ch == curses.KEY_RESIZE:
                    continue
                if ch in ("\n", "\r", curses.KEY_ENTER):
                    return "".join(buf)
                if ch == KEY_ESC:
                    return None
                if ch in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                    if pos > 0:
                        del buf[pos - 1]
                        pos -= 1
                elif ch == curses.KEY_DC:
                    if pos < len(buf):
                        del buf[pos]
                elif ch == curses.KEY_LEFT:
                    pos = max(0, pos - 1)
                elif ch == curses.KEY_RIGHT:
                    pos = min(len(buf), pos + 1)
                elif ch in (curses.KEY_HOME, "\x01"):
                    pos = 0
                elif ch in (curses.KEY_END, "\x05"):
                    pos = len(buf)
                elif ch == "\x15":
                    buf, pos = [], 0
                elif isinstance(ch, str) and ch.isprintable():
                    buf.insert(pos, ch)
                    pos += 1
        finally:
            curses.curs_set(0)

    def md_pager(self, title, md, head=None, live=None, keys=(), hint="", follow=False):
        """Muestra Markdown formateado (tablas con bordes, títulos y porcentajes en color).
        head(width) -> lines of (text, curses attr) drawn above the Markdown (e.g. a benchmark card's header).
        live() -> (token, busy): content that is still loading; re-rendered when the token changes, polled while busy.
        keys: extra keys that close the pager and are returned (e.g. n/p to move to the next item); hint: their help.
        follow: stay at the end while live content grows (like tail -f); scrolling up stops it, G resumes it."""
        styles = {"h1": curses.color_pair(C_ACC) | curses.A_BOLD, "h2": curses.color_pair(C_ACC) | curses.A_BOLD, "h3": curses.A_BOLD,
                  "bold": curses.A_BOLD, "dim": curses.color_pair(C_DIM), "code": curses.color_pair(C_WARN), "quote": curses.color_pair(C_DIM),
                  "border": curses.color_pair(C_DIM), "thead": curses.color_pair(C_ACC) | curses.A_BOLD, "good": curses.color_pair(C_OK) | curses.A_BOLD,
                  "mid": curses.color_pair(C_WARN) | curses.A_BOLD, "warn": curses.color_pair(C_MAG) | curses.A_BOLD,
                  "bad": curses.color_pair(C_BAD) | curses.A_BOLD, "normal": 0}
        md = _home_short(md)                     # paths as ~/… (the pager draws char by char, so put() can't shorten them)
        top = left = 0
        cache_w, lines, first, drawn = None, [], True, None
        token, busy = (live() if live else (None, False))
        while True:
            h, w = self.size()
            if live:
                tok, busy = live()
                if tok != token:
                    token, cache_w, first = tok, None, True
            if cache_w != w:
                lines, cache_w = (head(max(40, w - 4)) if head else []) + mdview.render(md, max(40, w - 4)), w
                if follow:
                    top = max(0, len(lines) - (h - 3))
            body = h - 3
            # first frame, or a jump of more than one line (PgDn, g/G, ←→): full repaint. curses' scroll optimisation can
            # otherwise leave rows of the previous page on screen
            if first or drawn is None or abs(top - drawn[0]) > 1 or left != drawn[1]:
                self.scr.clear()
                first = False
            else:
                self.scr.erase()
            drawn = (top, left)
            self.header(title() if callable(title) else title)
            for i in range(body):
                idx = top + i
                if idx >= len(lines):
                    break
                x, col = 1, 0
                for text, style in lines[idx]:
                    for ch in text:
                        cw = mdview.dwidth(ch)
                        if col >= left and x + cw < w:
                            self.put(2 + i, x, ch, style if isinstance(style, int) else styles.get(style, 0))
                            x += cw
                        col += cw
            more = max(0, len(lines) - body)
            self.footer((hint + " · " if hint else "") + T("↑↓ PgUp PgDn desplazar · ←→ horizontal · g/G inicio/fin · q salir",
                                                             "↑↓ PgUp PgDn scroll · ←→ horizontal · g/G top/end · q back")
                        + f"   [{min(top + 1, len(lines))}/{len(lines)}]")
            self.scr.refresh()
            k = self.key(120 if busy else -1)
            if k is None:
                continue
            if k in keys:
                return k
            if k in ("q", "Q", KEY_ESC):
                return
            if k in (curses.KEY_DOWN, "j"):
                top = min(more, top + 1)
            elif k in (curses.KEY_UP, "k"):
                top, follow = max(0, top - 1), False
            elif k in (curses.KEY_NPAGE, " "):
                top = min(more, top + body - 1)
            elif k == curses.KEY_PPAGE:
                top, follow = max(0, top - body + 1), False
            elif k == curses.KEY_RIGHT:
                left += 10
            elif k == curses.KEY_LEFT:
                left = max(0, left - 10)
            elif k in ("g", curses.KEY_HOME):
                top = left = 0
                follow = False
            elif k in ("G", curses.KEY_END):
                top = more
                follow = bool(live)

    def pager(self, title, text, footer="↑↓ PgUp PgDn desplazar · ←→ horizontal · g/G inicio/fin · q salir"):
        lines = text.split("\n")
        top = left = 0
        first = True
        while True:
            h, w = self.size()
            body = h - 3
            if first:                   # primer cuadro: repintado completo
                self.scr.clear()
                first = False
            else:
                self.scr.erase()
            self.header(title)
            for i in range(body):
                idx = top + i
                if idx >= len(lines):
                    break
                l = lines[idx]
                attr = 0
                if l.startswith("#"):
                    attr = curses.A_BOLD | curses.color_pair(C_ACC)
                elif "FAIL" in l[:8] or l.startswith("| `") and False:
                    attr = curses.color_pair(C_BAD)
                self.put(2 + i, 0, l[left:left + w - 1], attr)
            self.footer(f"{footer}   [{top + 1}/{len(lines)}]")
            self.scr.refresh()
            k = self.key()
            if k in ("q", "Q", KEY_ESC):
                return
            if k in (curses.KEY_DOWN, "j"):
                top = min(max(0, len(lines) - body), top + 1)
            elif k in (curses.KEY_UP, "k"):
                top = max(0, top - 1)
            elif k in (curses.KEY_NPAGE, " "):
                top = min(max(0, len(lines) - body), top + body - 1)
            elif k == curses.KEY_PPAGE:
                top = max(0, top - body + 1)
            elif k == curses.KEY_RIGHT:
                left += 8
            elif k == curses.KEY_LEFT:
                left = max(0, left - 8)
            elif k in ("g", curses.KEY_HOME):
                top = left = 0
            elif k in ("G", curses.KEY_END):
                top = max(0, len(lines) - body)

    def menu(self, title, items, footer="↑↓ mover · Enter elegir · q volver", sel=0, sub=""):
        """items: [(etiqueta, ayuda)]. Devuelve el índice elegido o None."""
        while True:
            h, w = self.size()
            self.scr.erase()
            self.header(title, sub)
            top = max(0, sel - (h - 6) + 1)
            rw = min(w - 2, max([len(_(l)) for l, _h in items] + [30]) + 6)
            for i, (label, hint) in enumerate(items[top:top + h - 6]):
                idx = top + i
                y = 3 + i
                if idx == sel:
                    self.sel_bar(y, 1, rw)
                    self.put(y, 3, _(label), self.color(255, self.SEL_BG, True), maxw=rw - 3)
                    self.hintline(hint)
                else:
                    self.put(y, 3, _(label), maxw=rw - 3)
            if len(items) > h - 6:                      # scroll position
                self.put(2, rw - 8, f"{sel + 1}/{len(items)}", self.color(self.MUTED))
            self.footer(footer)
            self.scr.refresh()
            k = self.key()
            if k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(items)
            elif k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(items)
            elif k in ("\n", "\r", curses.KEY_ENTER):
                return sel
            elif k in ("q", "Q", KEY_ESC):
                return None

    def form(self, title, fields, state, footer="", sub="", on_change=None, extra_keys=None, tick=None):
        """Formulario. Devuelve ('action', clave) o ('back', None). Modifica `fields[i].value`.
        tick(fields): called before every frame and every ~0.3 s, to show results that arrive in the background."""
        sel = state.get("sel", 0)
        sel_idx = [i for i, f in enumerate(fields) if f.kind in SELECTABLE]
        if not sel_idx:
            return "back", None
        if sel not in sel_idx:
            sel = sel_idx[0]
        note = ""
        while True:
            if state.get("rebuild"):
                state["sel"] = sel
                self.scr.clear()            # the rebuilt form has other rows: full repaint, or curses leaves ghosts
                return "rebuild", None
            if tick:
                tick(fields)
            h, w = self.size()
            self.scr.erase()
            self.header(title, sub)
            body = h - 6
            top = state.get("top", 0)
            if sel < top:
                top = sel
            if sel >= top + body:
                top = sel - body + 1
            # incluye la sección anterior si está justo encima
            state["top"] = top
            lw = min(max([len(_(f.label)) for f in fields if f.kind in ('text', 'int', 'float', 'bool', 'choice')] + [18]) + 2, max(18, w // 2 - 4))
            for i, f in enumerate(fields[top:top + body]):
                idx = top + i
                y = 3 + i
                if f.kind == "section":
                    if f.label:
                        self.section(y, 1, _(f.label), min(w - 3, 108))
                    continue
                if f.kind == "info":
                    self.put(y, 3, _(f.label) if f.value in (None, "") else f"{_(f.label)}{_(f.value)}", curses.color_pair(C_WARN) if f.hint == "warn" else curses.color_pair(C_DIM))
                    continue
                cur = idx == sel
                bg = self.SEL_BG if cur else None
                rowc = self.color(255 if cur else 252, bg, cur)
                if cur:
                    self.sel_bar(y, 1, min(w - 2, max(lw + 40, 70)))
                if f.kind == "action":
                    go = f.label.startswith("▶")
                    self.put(y, 3, "❯ " if not go else "", self.color(209, bg, True))
                    self.put(y, 3 if go else 5, _(f.label), self.color(78, bg, True) if go else rowc)
                    if f.value not in (None, ""):
                        self.put(y, (3 if go else 5) + len(_(f.label)) + 2, _(f.value), self.color(self.MUTED, bg))
                    continue
                self.put(y, 3, _(f.label)[:lw].ljust(lw), rowc)
                v = f.value
                if f.kind == "bool":
                    self.put(y, 3 + lw, "◉" if v else "○", self.color(78 if v else self.MUTED, bg, True))
                    self.put(y, 5 + lw, _("sí") if v and i18n.get_lang() == "es" else ("yes" if v else ("no")), self.color(78 if v else self.MUTED, bg))
                    continue
                if f.kind == "choice":
                    disp = _(dict(f.choices).get(v, v))
                    self.put(y, 3 + lw, "‹ ", self.color(self.MUTED, bg))
                    self.put(y, 5 + lw, disp, self.color(209, bg, True))
                    self.put(y, 5 + lw + len(disp), " ›", self.color(self.MUTED, bg))
                    continue
                if v in (None, ""):
                    self.put(y, 3 + lw, _(f.placeholder or "(vacío)"), self.color(self.MUTED, bg) | _italic())
                else:
                    self.put(y, 3 + lw, "•" * min(12, len(str(v))) if f.secret else str(v), self.color(255 if cur else 252, bg, cur))
            if sel < len(fields) and fields[sel].hint:
                self.hintline(fields[sel].hint)
            if note:
                self.toast(note, C_WARN)
            self.footer(footer or "↑↓ mover · Enter editar/activar · ←→ cambiar opción · Espacio marcar · q/Esc volver")
            self.scr.refresh()
            k = self.key(300 if tick else -1)
            if k is None:
                continue
            note = ""
            f = fields[sel]
            pos = sel_idx.index(sel)
            if extra_keys and k in extra_keys:
                extra_keys[k](f)
            elif k in (curses.KEY_DOWN, "j"):
                sel = sel_idx[(pos + 1) % len(sel_idx)]
            elif k in (curses.KEY_UP, "k"):
                sel = sel_idx[(pos - 1) % len(sel_idx)]
            elif k in ("q", "Q", KEY_ESC) and f.kind != "text":
                state["sel"] = sel
                return "back", None
            elif k == KEY_ESC:
                state["sel"] = sel
                return "back", None
            elif k in (curses.KEY_LEFT, curses.KEY_RIGHT) and f.kind == "choice" and f.choices:
                vals = [c[0] for c in f.choices]
                i = vals.index(f.value) if f.value in vals else 0
                f.value = vals[(i + (1 if k == curses.KEY_RIGHT else -1)) % len(vals)]
                if on_change:
                    on_change(f.key, fields)
            elif k in ("\n", "\r", curses.KEY_ENTER, " "):
                if f.kind == "bool":
                    f.value = not f.value
                    if on_change:
                        on_change(f.key, fields)
                elif f.kind == "choice" and f.choices:
                    vals = [c[0] for c in f.choices]
                    i = vals.index(f.value) if f.value in vals else -1
                    f.value = vals[(i + 1) % len(vals)]
                    if on_change:
                        on_change(f.key, fields)
                elif f.kind == "action":
                    state["sel"] = sel
                    return "action", f.key
                elif f.kind in ("text", "int", "float") and k != " ":
                    new = self.edit(f.label, "" if f.value is None else str(f.value), secret=f.secret, hint=f.hint)
                    if new is not None:
                        try:
                            f.value = int(new) if f.kind == "int" else float(new) if f.kind == "float" else new
                            if on_change:
                                on_change(f.key, fields)
                        except ValueError:
                            note = _(f"Valor no válido para «{f.label}» (se esperaba {'un entero' if f.kind == 'int' else 'un número'})")


# ---------------------------------------------------------------------------------------------
_status_cache = {"t": 0.0, "text": ""}


def running_status() -> str:
    """«● slmbench en marcha: modelo 12/40» si hay ejecuciones activas (comprobado como mucho cada segundo)."""
    now = time.time()
    if now - _status_cache["t"] < 1.0:
        return _status_cache["text"]
    runs = engine.active_runs()
    text = ""
    if runs:
        a = runs[0]
        prog = f" {a['done']}/{a['total']}" if a.get("total") else f" {a['done']}"
        more = f" (+{len(runs) - 1})" if len(runs) > 1 else ""
        text = f"● {T('en marcha', 'running')}: {a['model'][:28]}{prog}{more} · {T('clic/^R ver', 'click/^R view')}"
    _status_cache.update(t=now, text=text)
    return text


def sync(fields, s):
    """Copia los valores del formulario a Settings."""
    for f in fields:
        if f.kind in ("text", "int", "float", "bool", "choice") and hasattr(s, f.key):
            setattr(s, f.key, f.value)


# ---------------------------------------------------------------------------------------------
class App:
    def __init__(self, scr, settings):
        self.ui = UI(scr)
        self.ui.on_status = self.screen_running
        self.quip = random.choice(art.QUIPS)     # frase graciosa de la pantalla de inicio
        self.s = settings
        self.status = ("…", C_DIM)
        self.models_seen: list[str] = []
        self.connected = False
        self.needs_key = False
        self.meta: dict = {}
        self._meta_cache: dict = {}
        self._probe_lock = threading.Lock()
        self._probing = False
        self._probed_at = 0.0
        self.status_gen = 0                       # +1 each time the server status is updated (screens redraw on change)
        self._prewarmed = threading.Event()
        self.jobs: list[dict] = []                # runs left going in the background (live_task → back to the menu)
        threading.Thread(target=self._prewarm, daemon=True).start()

    # ---- estado del modelo a evaluar ----
    def refresh_status(self, wait: bool = True):
        """Asks the server which models it serves (and llama-server's metadata). wait=False: in a background thread, so a
        screen never waits for the network; the screens poll `status_gen` and redraw when the answer arrives."""
        if not wait:
            with self._probe_lock:
                if self._probing:
                    return
                self._probing = True
            threading.Thread(target=self._probe, daemon=True).start()
            return
        self._probe()

    def check_server(self, fresh: float = 0.0):
        """Asks the server now, with a spinner instead of a frozen screen (Esc stops waiting). fresh = seconds an earlier
        answer is still good for (0 = always ask)."""
        if fresh and time.time() - self._probed_at < fresh and not self._probing:
            return
        self.busy(T("Servidor", "Server"), T(f"Consultando {self.s.base_url}…", f"Asking {self.s.base_url}…"), self._probe)

    def _probe(self):
        s = self.s
        meta, needs_key, models = {}, False, []
        try:
            models = servers.probe(s.base_url, s.api_key, timeout=1.5)
            connected = True
            if s.model and s.model not in models:
                served = ", ".join(models[:3]) + ("…" if len(models) > 3 else "")
                status = (T(f"⚠ el servidor sirve «{served}», no «{s.model}» · pulsa 1 para cambiarlo",
                            f"⚠ the server serves \"{served}\", not \"{s.model}\" · press 1 to switch")
                          if models else T(f"⚠ el servidor no ofrece «{s.model}»", f"⚠ the server does not offer \"{s.model}\""), C_WARN)
            else:
                status = (T(f"● conectado · {len(models)} modelo(s) en el servidor", f"● connected · {len(models)} model(s) on the server"), C_OK)
            mid = s.model or (models[0] if models else "")
            key = (s.base_url, mid)
            if key in self._meta_cache:                # the metadata of a model doesn't change while it is served
                meta = self._meta_cache[key]
            else:
                try:
                    meta = servers.model_meta(s.base_url, mid, api_key=s.api_key)
                    self._meta_cache[key] = meta
                except Exception:
                    meta = {}
        except Exception as e:
            connected = False
            needs_key = servers.needs_key(e)
            self._meta_cache.clear()
            status = (("○ " + servers.probe_error(e, s.base_url, bool(s.api_key)) + " · " +
                       T("ponla en «Clave de API»", "set it in 'API key'")) if needs_key else
                      T(f"○ sin conexión ({type(e).__name__}) · elige el modelo en «Modelo a evaluar»",
                        f"○ not connected ({type(e).__name__}) · pick the model in \"Model to evaluate\""), C_BAD)
        self.models_seen, self.connected, self.needs_key, self.meta, self.status = models, connected, needs_key, meta, status
        self._probed_at = time.time()
        with self._probe_lock:
            self._probing = False
            self.status_gen += 1

    def _prewarm(self):
        """Slow checks done once, in the background, before a screen needs them: whether the sandbox is ready (≈ 1 s of
        `sbx` calls) is what the coding benchmarks' lists show."""
        try:
            code = [k for k, v in cat.catalog().items() if v["mode"] == "code" and v.get("requires")]
            if code:
                cat.missing_tools(code[0])      # the sandbox status + the compilers inside it, cached for the session
        except Exception:
            pass
        try:                                    # sandboxes left by a slmbench that died (kill -9, a crash): removed now
            from . import sandbox
            if sandbox.sbx_installed():
                sandbox.prune()
        except Exception:
            pass
        finally:
            self._prewarmed.set()

    def splash(self):
        """The loading screen: the wordmark revealed with a light sweep in the logo's gradient, the benchmark bars dancing,
        the tagline typed out, and the real start-up work with a spinner each (the model server, the sandboxes, the
        leaderboard). As long as that work takes (1.6 s at least, 5 s at most); any key skips it."""
        ui = self.ui
        h, w = ui.size()
        if (not getattr(self.s, "splash", True) or os.environ.get("SLMBENCH_NO_SPLASH") or ASCII_ONLY or h < 20 or w < 60
                or curses.COLORS < 256):
            return
        import math
        steps = [(T("Conectando con el servidor del modelo", "Connecting to the model server"), lambda: (self._probe(), self.connected)[1]),
                 (T("Comprobando los sandboxes", "Checking the sandboxes"), lambda: self._prewarmed.wait(6)),
                 (T("Cargando el leaderboard", "Loading the leaderboard"), lambda: bool(leaderboard.load_entries(self.s)) or True)]
        done: list = [None] * len(steps)

        def run(i, fn):
            try:
                done[i] = bool(fn())
            except Exception:  # noqa: BLE001 - the splash never stops slmbench from opening
                done[i] = False
        for i, (_l, fn) in enumerate(steps):
            threading.Thread(target=run, args=(i, fn), daemon=True).start()
        G = art.BAR_GRADIENT
        t0, MIN, MAX, held = time.time(), 1.6, 5.0, None
        while True:
            h, w = ui.size()
            scale = 2 if w >= art.wordmark_width(scale=2) + 6 and h >= 30 else 1
            rows, ww = art.wordmark(scale=scale), art.wordmark_width(scale=scale)
            total = len(rows) + 9 + len(steps)
            y0, x0 = max(1, (h - total) // 2), max(0, (w - ww) // 2)
            t = time.time() - t0
            ui.scr.erase()
            # the wordmark, revealed left to right; the edge of the reveal and a later sweep glow in the gradient
            reveal = ww * min(1.0, t / 0.75)
            sweep = (t - 0.9) / 0.7 * (ww + 16) - 8 if t > 0.9 else -99
            for i, segs in enumerate(rows):
                xx = 0
                for text, key in segs:
                    fg, bold = art.PALETTE[key]
                    for j, ch in enumerate(text):
                        c = xx + j
                        if ch == " " or c >= reveal:
                            continue
                        edge = reveal - c < 5 and t < 0.85
                        glow = abs(c - sweep) < 3
                        col = G[(c + int(t * 18)) % len(G)] if edge else (231 if glow else fg)
                        ui.put(y0 + i, x0 + c, ch, ui.color(col, None, bold or glow))
                    xx += len(text)
            # the benchmark bars, growing then dancing
            by = y0 + len(rows) + 1
            grow = min(1.0, t / 0.6)
            levels = "▁▂▃▄▅▆▇█"
            for x in range(ww):
                v = (0.5 + 0.3 * math.sin(x * 0.55 + t * 5.5) + 0.2 * math.sin(x * 1.7 - t * 3.7)) * grow
                ui.put(by, x0 + x, levels[max(0, min(7, int(v * 8)))], ui.color(G[int(x * (len(G) - 1) / max(1, ww - 1))]))
            # the tagline, typed out
            tag = T(art.TAGLINE_ES, art.TAGLINE) if hasattr(art, "TAGLINE_ES") else art.TAGLINE
            n = int(len(tag) * min(1.0, max(0.0, (t - 0.45) / 0.8)))
            ui.put(by + 2, max(0, (w - len(tag)) // 2), tag[:n] + ("▌" if n < len(tag) and int(t * 6) % 2 == 0 else ""),
                   ui.color(250) | _italic())
            # the real start-up work
            lw = max(len(l) for l, _f in steps) + 4
            sx, sy = max(0, (w - lw) // 2), by + 4
            for i, (label, _f) in enumerate(steps):
                st = done[i]
                if st is None:
                    mark, col = SPIN[int(t * 12) % len(SPIN)], 209
                elif st:
                    mark, col = "✔", 78
                else:
                    mark, col = "○", 220
                ui.put(sy + i, sx, mark, ui.color(col, None, True))
                ui.put(sy + i, sx + 2, label, ui.color(255 if st is not None else 250))
            bw = min(44, w - 8)
            frac = min(1.0, 0.75 * sum(d is not None for d in done) / len(done) + 0.25 * min(1.0, t / MIN))
            px = max(0, (w - bw) // 2)
            for x in range(bw):
                filled = x < int(bw * frac)
                ui.put(sy + len(steps) + 1, px + x, "━" if filled else "─",
                       ui.color(G[int(x * (len(G) - 1) / max(1, bw - 1))] if filled else 238))
            hint = T("cualquier tecla para saltar", "any key to skip")
            ui.put(h - 2, max(0, (w - len(hint)) // 2), hint, ui.color(240))
            ui.scr.refresh()
            finished = all(d is not None for d in done) and t >= MIN
            if finished and held is None:
                held = time.time()
            if (held and time.time() - held > 0.35) or t >= MAX:
                break
            k = ui.key(33)
            if k is not None and k != curses.KEY_RESIZE:
                break
        ui.scr.clear()                              # full repaint next: curses would otherwise leave splash cells behind
        ui.scr.refresh()

    def current_model(self) -> str:
        return self.s.model or (self.models_seen[0] if self.models_seen else "")

    def model_panel(self, y: int, x: int = 2, w: int | None = None) -> int:
        """Caja «MODELO A EVALUAR». Devuelve la siguiente fila libre."""
        ui = self.ui
        W = (w or ui.size()[1] - 4)
        W = max(40, min(W, 96))
        name = self.current_model() or "— ninguno —"
        m = self.meta or {}
        bits = [b for b in (f"{m['params_b']:g}B params" if m.get("params_b") else "", m.get("quant", ""),
                            f"{m['size_gb']} GB" if m.get("size_gb") else "", m.get("backend", "")) if b]
        if self.current_model():
            _eid, base, q = leaderboard.entry_identity(self.current_model(), m.get("quant"))
            label = f"{base} · {q}" if q else base
        else:
            label = "-"
        acc = curses.color_pair(C_ACC)
        ui.box(y, x, 7, W, _("MODELO A EVALUAR"), acc)
        ui.put(y + 1, x + 2, name[: W - 4], curses.A_BOLD)
        ui.put(y + 2, x + 2, (" · ".join(bits) or _("(sin metadatos)"))[: W - 4], curses.color_pair(C_DIM))
        ui.put(y + 3, x + 2, _(f"servidor {self.s.base_url}  ·  fila en el leaderboard: {label}")[: W - 4], curses.color_pair(C_DIM))
        stxt = _(self.status[0])
        if self._probing and self.status[0] == "…":
            stxt = SPIN[int(time.time() * 8) % len(SPIN)] + " " + T("conectando…", "connecting…")
        mtxt, mattr = self.memory_line()
        ui.put(y + 4, x + 2, T("memoria ", "memory ") + " ", curses.color_pair(C_DIM))
        ui.put(y + 4, x + 2 + len(T("memoria ", "memory ")) + 1, mtxt[: W - 14], mattr)
        ui.put(y + 5, x + 2, stxt[: W - 4], curses.color_pair(self.status[1]) | curses.A_BOLD)
        return y + 7

    def memory_line(self) -> tuple[str, int]:
        """The home panel's «memory» line: hf-mem's estimate for the model under test (computed in the background)."""
        from . import memory
        guess = self.memory_target()
        if not guess:
            return (T("— (pulsa 8 e indica el repo de Hugging Face)", "— (press 8 and give the Hugging Face repo)") if self.current_model()
                    else "—"), curses.color_pair(C_DIM)
        if not memory.command():
            return T("hf-mem no está instalado", "hf-mem is not installed"), curses.color_pair(C_DIM)
        r = memory.estimate_async(guess[0], guess[1], fallbacks=guess[2])
        if r is None:
            return SPIN[int(time.time() * 8) % len(SPIN)] + T(" estimando con hf-mem…", " estimating with hf-mem…"), curses.color_pair(C_DIM)
        if r.get("error"):
            return "⚠ " + r["error"], curses.color_pair(C_WARN)
        line = memory.summary(r)
        f = "" if r.get("files") else memory.fit(memory.parts(r).get("total"))
        mark = {"ok": T("   ✔ cabe", "   ✔ fits"), "tight": T("   ⚠ justo", "   ⚠ tight"), "no": T("   ✗ no cabe", "   ✗ does not fit")}.get(f, "")
        return line + mark, curses.color_pair({"tight": C_WARN, "no": C_BAD}.get(f, C_OK)) | curses.A_BOLD

    def memory_pending(self) -> bool:
        from . import memory
        guess = self.memory_target()
        return bool(guess and memory.command() and memory.estimate_async(guess[0], guess[1], fallbacks=guess[2]) is None)

    def screen_memory_current(self):
        """Main menu «8»: the estimate for the model under test straight away; asks only if it can't be told."""
        guess = self.memory_target()
        if guess:
            self.memory_view(guess[0], guess[1], fallbacks=guess[2])
        else:
            self.screen_memory()

    # ---- menú principal ----
    def main(self):
        sel = 0
        self.refresh_status(wait=False)
        def menu_items():                          # rebuilt on every frame: «L» switches the language at once
            return [
                ("1", T("Modelo a evaluar", "Model to evaluate"), T("Elige qué modelo se mide: un GGUF local (arranca llama-server) o un servidor ya en marcha (llama-server, Ollama, LM Studio, vLLM, Unsloth Studio)",
                                                                   "Choose the model to measure: a local GGUF (starts llama-server) or a running server (llama-server, Ollama, LM Studio, vLLM, Unsloth Studio)")),
                ("2", T("Lanzar benchmarks", "Run benchmarks"), T("Modo general o programación; benchmarks públicos vía lighteval con progreso en vivo",
                                                                   "General or coding mode; public benchmarks via lighteval with live progress")),
                ("3", T("Benchmarks", "Benchmarks"), T("Qué mide cada benchmark y una tarea real de muestra", "What each benchmark measures and a real sample task")),
                ("4", T("Resultados y comparación", "Results & comparison"), T("Informes de ejecuciones anteriores y comparación lado a lado", "Reports of previous runs and side-by-side comparison")),
                ("5", "Leaderboard", T("Clasificación por modelo y quant (leaderboard.md / .html)", "Ranking by model and quant (leaderboard.md / .html)")),
                ("6", T("Ajustes", "Settings"), T("Temperatura, paralelismo, timeout, carpeta de resultados, idioma", "Temperature, parallelism, timeout, results folder, language")),
                ("7", T("Chatear con el modelo", "Chat with the model"), T("Pruébalo a mano en una conversación con llama cli (GGUF local)",
                                                                           "Try it by hand in a conversation with llama cli (local GGUF)")),
                ("8", T("Memoria del modelo (hf-mem)", "Model memory (hf-mem)"), T("Cuánta memoria necesita el modelo a evaluar (pesos + KV cache + overhead) y si cabe en este equipo, con hf-mem",
                                                                 "How much memory the model under test needs (weights + KV cache + overhead) and whether it fits on this machine, with hf-mem")),
                ("9", T("Trazas del agente", "Agent traces"), T("Cada paso del modelo como agente (modo agente: mini-SWE-agent, Pi, OpenCode): razonamiento, respuesta, comando o tool call y su salida",
                                                                 "Every step of the model as an agent (agent mode: mini-SWE-agent, Pi, OpenCode): reasoning, answer, command or tool call and its output")),
                ("q", T("Salir", "Quit"), ""),
            ]

        actions = [self.screen_model, self.screen_run, self.screen_benchmarks, self.screen_results, self.screen_leaderboard, self.screen_config, self.screen_chat, self.screen_memory_current, self.screen_traces, None]
        while True:
            items = menu_items()
            ui = self.ui
            h, w = ui.size()
            ui.scr.erase()
            ui.header(T("evaluación de modelos de lenguaje pequeños", "small language model evaluation"), "")
            y = 2
            panel_w = None
            quip = "» " + T(self.quip[1], self.quip[0])
            info = f"v{__version__}  ·  " + T("modelos pequeños · benchmarks públicos · basado en lighteval",
                                                "small models · public benchmarks · built on lighteval")
            center = lambda text_w: max(2, (w - text_w) // 2)
            # the wordmark only where everything else still fits: panel (7) + server line + the menu + help and footer (3)
            below = 7 + 1 + 1 + len(items) + 3
            big_h, small_h = len(art.wordmark(scale=2)) + 7, len(art.wordmark()) + 4
            if 2 + big_h + below <= h and w >= art.wordmark_width(scale=2) + 4:      # wordmark 2x + bars + tagline
                ww = art.wordmark_width(scale=2)
                x = center(ww)
                rows = ui.draw_wordmark(y + 1, x, scale=2)
                ui.draw_bars(y + 1 + rows, x, ww)
                ui.put(y + rows + 3, center(len(art.TAGLINE)), art.TAGLINE, curses.A_BOLD)
                ui.put(y + rows + 4, center(len(info)), info, curses.color_pair(C_DIM))
                ui.put(y + rows + 5, center(len(quip)), quip, curses.color_pair(C_MAG) | _italic())
                y += rows + 7
            elif 2 + small_h + below <= h and w >= art.wordmark_width() + 4:     # wordmark 1x + bars
                ww = art.wordmark_width()
                x = center(ww)
                rows = ui.draw_wordmark(y, x)
                ui.draw_bars(y + rows, x, ww)
                ui.put(y + rows + 1, center(len(art.TAGLINE)), art.TAGLINE, curses.A_BOLD)
                ui.put(y + rows + 2, center(len(quip)), quip, curses.color_pair(C_MAG) | _italic())
                y += rows + 4
            y = self.model_panel(y, w=panel_w)
            st = servers.managed_state()
            if st:
                ui.put(y, 4, _(f"llama-server lanzado desde aquí: PID {st['pid']} · puerto {st['port']}"), curses.color_pair(C_DIM))
            y += 1
            for i, (key, label, hint) in enumerate(items):
                yy = y + 1 + i
                cur = i == sel
                bg = ui.SEL_BG if cur else None
                if cur:
                    ui.sel_bar(yy, 2, 44)
                    ui.hintline(hint)
                ui.put(yy, 4, f" {key} ", ui.color(ui.INK, 209, True) if cur else ui.color(209, 236, True))
                ui.put(yy, 8, label, ui.color(255, bg, True) if cur else (curses.A_BOLD if i == 0 and not self.connected else 0))
            ui.footer(T("↑↓ mover · Enter o 1-9 elegir · L idioma (EN/ES) · r reconectar · q salir", "↑↓ move · Enter or 1-9 select · L language (EN/ES) · r reconnect · q quit"))
            ui.scr.refresh()
            self.notify_jobs()                         # a run left in the background has ended: say so here
            k = ui.key(250 if self._probing or self.memory_pending() else 1000 if self._jobs_alive() else 1500)
            choice = None
            if k in (curses.KEY_DOWN, "j"):
                sel = (sel + 1) % len(items)
            elif k in (curses.KEY_UP, "k"):
                sel = (sel - 1) % len(items)
            elif k in ("r", "R"):
                self.refresh_status(wait=False)
            elif k in ("l", "L"):
                self.s.ui_lang = "es" if self.s.ui_lang == "en" else "en"
                i18n.set_lang(self.s.ui_lang)
                config.save(self.s)
                self.refresh_status(wait=False)
            elif k in ("q", "Q", KEY_ESC):
                if self.quit():
                    return
            elif isinstance(k, str) and k in "123456789" and k:
                choice = int(k) - 1
            elif k in ("\n", "\r", curses.KEY_ENTER):
                choice = sel
            if choice is not None:
                sel = choice
                fn = actions[choice]
                if fn is None:
                    if self.quit():
                        return
                else:
                    fn()
                self.refresh_status(wait=False)

    def _jobs_alive(self) -> list[dict]:
        return [j for j in self.jobs if not j["done"]]

    def notify_jobs(self):
        """Runs left in the background that have ended since the last look: one message each (report or error)."""
        for j in self.jobs:
            if j["done"] and not j.get("notified"):
                j["notified"] = True
                if j.get("err"):
                    self.ui.message(f"{j['title']}\n\nError: {j['err']}", C_BAD, T("Ejecución terminada", "Run finished"))
                elif j.get("res"):
                    res = Path(str(j["res"]))
                    if (res / "report.md").exists() and self.ui.confirm(
                            T(f"✔ {j['title']}: terminada{' (cancelada)' if j.get('cancelled') else ''}.\n\n¿Ver el informe? (también en 4 · Resultados)",
                              f"✔ {j['title']}: finished{' (cancelled)' if j.get('cancelled') else ''}.\n\nOpen the report? (also in 4 · Results)"), default=True):
                        self.ui.md_pager(f"{T('Informe', 'Report')} · {res.name}", (res / "report.md").read_text())

    def quit(self):
        alive = self._jobs_alive()
        if alive:
            if not self.ui.confirm(T(f"Hay {len(alive)} ejecución(es) en marcha en esta TUI. Si sales se cancelan (se guarda lo terminado "
                                     "y se borran sus sandboxes). ¿Salir?",
                                     f"{len(alive)} run(s) are still going in this TUI. Quitting cancels them (finished results are kept "
                                     "and their sandboxes removed). Quit?")):
                return False
            ui = self.ui
            for j in alive:
                j["cancel"].set()
            t0 = time.time()
            while any(j["thread"].is_alive() for j in alive) and time.time() - t0 < 120:
                ui.scr.erase()
                ui.header(T("Saliendo", "Quitting"))
                ui.put(3, 2, f"{SPIN[int(time.time() * 10) % len(SPIN)]} " + T("Cancelando las ejecuciones en marcha…", "Cancelling the running runs…"),
                       curses.color_pair(C_WARN) | curses.A_BOLD)
                ui.scr.refresh()
                time.sleep(0.1)
        st = servers.managed_state()
        if st and self.ui.confirm(f"El llama-server lanzado desde aquí sigue en marcha (PID {st['pid']}). ¿Detenerlo al salir?", default=True):
            ui = self.ui
            ui.scr.erase()
            ui.header(T("Saliendo", "Quitting"))
            ui.put(3, 2, T(f"Deteniendo llama-server ({st.get('alias') or 'modelo'}, PID {st['pid']}, puerto {st.get('port')})…",
                           f"Stopping llama-server ({st.get('alias') or 'model'}, PID {st['pid']}, port {st.get('port')})…"),
                   curses.color_pair(C_WARN) | curses.A_BOLD)
            ui.put(5, 2, T("Puede tardar unos segundos mientras libera la memoria del modelo.",
                           "This can take a few seconds while it frees the model's memory."), curses.color_pair(C_DIM))
            ui.scr.refresh()
            servers.stop()
            self.exit_note = T(f"✔ llama-server detenido (PID {st['pid']}).", f"✔ llama-server stopped (PID {st['pid']}).")
        elif st:
            self.exit_note = T(f"llama-server sigue en marcha (PID {st['pid']}, puerto {st.get('port')}).",
                               f"llama-server is still running (PID {st['pid']}, port {st.get('port')}).")
        config.save(self.s)
        return True

    # ---- modelo a evaluar ----
    PRESETS = [
        ("llama-server (llama.cpp)", "http://127.0.0.1:8080/v1"),
        ("Ollama", "http://127.0.0.1:11434/v1"),
        ("LM Studio", "http://127.0.0.1:1234/v1"),
        ("vLLM / SGLang", "http://127.0.0.1:8000/v1"),
        ("Unsloth Studio", "http://127.0.0.1:8888/v1"),
        ("Otra URL…", ""),
    ]
    STUDIO = "Unsloth Studio"

    def screen_model(self):
        ui, s = self.ui, self.s
        state = {}
        while True:
            self.refresh_status(wait=False)
            st = servers.managed_state()
            fields = [
                Field("", "¿Qué modelo quieres medir?", "section"),
                Field("", "slmbench habla con cualquier servidor con API OpenAI. Elige una de las dos formas:", "info", ""),
                Field("local", "A · Modelo local: elegir un .gguf y arrancar llama-server", "action", "(recomendado)"),
                Field("remote", "B · Conectar a un servidor que ya está en marcha", "action", "llama-server, Ollama, LM Studio, vLLM, Unsloth Studio…"),
                Field("", "Selección actual", "section"),
                Field("", "Servidor: ", "info", s.base_url),
                Field("api_key", "Clave de API (si el servidor la pide)", "text", s.api_key, secret=True,
                      hint=(T("Vacía: se usa UNSLOTH_API_KEY (Unsloth Studio). Una clave aquí tiene prioridad y se guarda en "
                              "~/.config/slmbench/config.json con permisos 600.",
                              "Empty: UNSLOTH_API_KEY is used (Unsloth Studio). A key here takes precedence and is saved in "
                              "~/.config/slmbench/config.json with permissions 600.")
                            if not s.api_key and os.environ.get("UNSLOTH_API_KEY") and servers.is_unsloth_studio(s.base_url) else
                            "Se guarda en ~/.config/slmbench/config.json con permisos 600."),
                      placeholder=("(UNSLOTH_API_KEY)" if not s.api_key and os.environ.get("UNSLOTH_API_KEY")
                                   and servers.is_unsloth_studio(s.base_url) else "")),
                Field("pick", "Modelo", "action", self.current_model() + ("" if s.model else "  (el primero del servidor)") if self.current_model() else "— sin conexión —"),
                Field("label", "Etiqueta de la ejecución (opcional)", "text", s.label,
                      hint="Nombre de la carpeta de resultados y de los informes de este modelo (al cambiar de modelo deja de aplicarse). El leaderboard usa siempre el id del modelo; para comparar variantes, sírvelas con ids distintos (--alias)."),
                Field("test", "Probar conexión", "action", ""),
                Field("status", "", "info", self.status[0], hint="" if self.connected else "warn"),
                Field("", T("Herramientas", "Tools"), "section"),
                Field("mem", T("Estimar la memoria de un modelo (hf-mem)", "Estimate a model's memory (hf-mem)"), "action", "",
                      hint=T("Pesos + KV cache + overhead de un repo de Hugging Face o un GGUF local, sin descargar los pesos",
                             "Weights + KV cache + overhead of a Hugging Face repo or a local GGUF, without downloading the weights")),
            ]
            seen_gen = [self.status_gen]

            def tick(fs):
                if self.status_gen != seen_gen[0]:          # the background probe answered: show it
                    seen_gen[0] = self.status_gen
                    for f in fs:
                        if f.key == "status":
                            f.value, f.hint = self.status[0], "" if self.connected else "warn"
                        elif f.key == "pick":
                            f.value = (self.current_model() + ("" if s.model else T("  (el primero del servidor)", "  (the server's first)"))
                                       if self.current_model() else T("— sin conexión —", "— not connected —"))
            if st:
                fields.insert(4, Field("", "", "info", f"llama-server lanzado desde aquí: PID {st['pid']} · puerto {st['port']} · {st['alias']}"))
            if servers.is_unsloth_studio(s.base_url):     # Studio loads the model (and its LoRA) itself: ask it from here
                from . import studio
                now = studio.status(s.base_url, servers.effective_key(s.base_url, s.api_key))
                at = next(i for i, f in enumerate(fields) if f.key == "status") + 1
                fields[at:at] = [
                    Field("", "Unsloth Studio", "section"),
                    Field("", "", "info", (T("Cargado: ", "Loaded: ") + now["model"] + (f" ({now['variant']})" if now.get("variant") else "")
                                           + (f" · ctx {now['context_length']}" if now.get("context_length") else "")
                                           + (f"  +  LoRA {now['lora']}" if now.get("lora") else "")) if now.get("model")
                          else T("Studio no tiene ningún modelo cargado.", "Studio has no model loaded.")),
                    Field("studio_pick", T("Modelo base (GGUF de Studio)", "Base model (Studio GGUF)"), "action",
                          s.studio_model or T("(elegir)", "(pick)"),
                          hint=T("Los modelos que Studio tiene descargados.", "The models Studio has downloaded.")),
                    Field("studio_variant", T("Cuantización", "Quantization"), "text", s.studio_variant,
                          hint=T("p. ej. Q8_0. Vacío = la que Studio tenga descargada.", "e.g. Q8_0. Empty = the one Studio has downloaded."),
                          placeholder=T("(la descargada)", "(the downloaded one)")),
                    Field("studio_lora", T("Adaptador LoRA (opcional)", "LoRA adapter (optional)"), "text", s.studio_lora,
                          hint=T("Un .gguf, un repo de HF (org/repo[:fichero.gguf]) o su URL; un repo PEFT usa su versión -GGUF. Se descarga "
                                 "aquí y Studio lo carga con --lora.",
                                 "A .gguf, a HF repo (org/repo[:file.gguf]) or its URL; a PEFT repo uses its -GGUF version. It's downloaded "
                                 "here and Studio loads it with --lora.")),
                    Field("studio_lora_scale", T("Escala del LoRA", "LoRA scale"), "float", s.studio_lora_scale),
                    Field("studio_ctx", T("Contexto (tokens)", "Context (tokens)"), "int", s.studio_ctx,
                          hint=T("p. ej. 32768 o 65536 para el modo agente. 0 = el que Studio elija según la memoria. Studio puede "
                                 "dar menos si no cabe.",
                                 "e.g. 32768, or 65536 for agent mode. 0 = the one Studio picks for the memory. Studio may give less "
                                 "if it doesn't fit.")),
                    Field("studio_load", "▶ " + T("Cargar en Unsloth Studio", "Load in Unsloth Studio"), "action", "",
                          hint=T("Studio carga el modelo (con el adaptador) y queda como el modelo a evaluar.",
                                 "Studio loads the model (with the adapter) and it becomes the model to evaluate.")),
                ]
            old_label = s.label
            kind, key = ui.form("Modelo a evaluar", fields, state, footer="↑↓ mover · Enter elegir/editar · q volver", tick=tick)
            sync(fields, s)
            if s.label != old_label:
                s.label_model = self.current_model()      # the label belongs to this model only
            config.save(s)
            if kind == "back":
                return
            if key == "local":
                self.screen_server()
            elif key == "studio_pick":
                from . import studio
                ms, err = self.busy("Unsloth Studio", T("Leyendo los modelos de Studio…", "Reading Studio's models…"),
                                    lambda: studio.models(s.base_url, servers.effective_key(s.base_url, s.api_key)))
                if err or not ms:
                    ui.message(str(err or T("Studio no tiene modelos descargados.", "Studio has no downloaded models.")), C_BAD, "Unsloth Studio")
                    continue
                i = ui.menu(T("Modelo base en Unsloth Studio", "Base model in Unsloth Studio"),
                            [(f"{m['id']:<44} {m['quant']:<12}" + (T(" ● cargado", " ● loaded") if m["loaded"] else ""), "") for m in ms],
                            sub=T("El adaptador LoRA se elige aparte, en «Adaptador LoRA»", "The LoRA adapter is set apart, in \"LoRA adapter\""))
                if i is not None:
                    s.studio_model, s.studio_variant = ms[i]["id"], ms[i]["quant"]
                    config.save(s)
            elif key == "studio_load":
                from . import studio
                label = s.studio_model + (f" + LoRA {s.studio_lora.strip()}" if s.studio_lora.strip() else "")
                res, err = self.busy("Unsloth Studio", T(f"Studio está cargando {label}…", f"Studio is loading {label}…"),
                                     lambda: studio.load(s.base_url, servers.effective_key(s.base_url, s.api_key), s.studio_model,
                                                         s.studio_variant, s.studio_lora, s.studio_lora_scale, s.studio_ctx))
                if err:
                    if str(err) != T("cancelado", "cancelled"):
                        ui.message(str(err), C_BAD, "Unsloth Studio")
                    continue
                s.model = res["model"]
                config.save(s)
                self.refresh_status(wait=True)
                ui.message(T(f"Cargado en Unsloth Studio: {res['model']}", f"Loaded in Unsloth Studio: {res['model']}")
                           + (f"\n+ LoRA {res['lora']}" if res["lora"] else "") + (f"\n{T('contexto', 'context')}: {res['context_length']}"
                                                                                  if res.get("context_length") else "")
                           + (T(f" (pediste {res['asked_ctx']}: Studio lo ajustó a la memoria)", f" (you asked for {res['asked_ctx']}: Studio "
                                "fitted it to the memory)") if res.get("asked_ctx") and res.get("context_length")
                              and res["context_length"] != res["asked_ctx"] else "")
                           + "\n\n" + T("Es el modelo a evaluar; en el leaderboard irá a la fila del adaptador.",
                                          "It's the model to evaluate; in the leaderboard it goes to the adapter's row."), C_OK, "Unsloth Studio")
            elif key == "remote":
                i = ui.menu("¿Qué servidor está en marcha?", [(f"{n:<26} {u}", "") for n, u in self.PRESETS],
                            sub="Arráncalo antes por tu cuenta (p. ej. `ollama serve`, LM Studio › Developer › Start server)")
                if i is None:
                    continue
                url = self.PRESETS[i][1]
                studio = self.PRESETS[i][0] == self.STUDIO
                if not url:
                    url = ui.edit("URL base (termina en /v1)", s.base_url) or ""
                if url:
                    s.base_url = url.strip()
                    s.model = ""
                    config.save(s)
                    if studio and not s.api_key:          # Unsloth Studio always wants a key: UNSLOTH_API_KEY, or ask for it
                        if os.environ.get("UNSLOTH_API_KEY"):
                            ui.toast(T("Unsloth Studio: se usa la clave de UNSLOTH_API_KEY", "Unsloth Studio: using the key in UNSLOTH_API_KEY"), C_OK)
                        else:
                            key_in = ui.edit(T("Clave de API de Unsloth Studio", "Unsloth Studio API key"), "", secret=True,
                                             hint=T("Unsloth Studio pide una clave (Ajustes › API). También puedes exportar UNSLOTH_API_KEY.",
                                                    "Unsloth Studio needs a key (Settings › API). You can also export UNSLOTH_API_KEY."))
                            if key_in:
                                s.api_key = key_in.strip()
                                config.save(s)
                    self.check_server()
                    while self.needs_key:                 # the server is there but wants a key: ask for it right here
                        key_in = ui.edit(T("Clave de API", "API key"), s.api_key, secret=True,
                                         hint=self.status[0].lstrip("○ ") + "  ·  Esc: " + T("cancelar", "cancel"))
                        if key_in is None:
                            break
                        s.api_key = key_in.strip()
                        config.save(s)
                        self.check_server()
                    if self.connected:
                        self.pick_model()
                    elif self.needs_key:
                        ui.message(self.status[0], C_BAD, T("Falta la clave de API", "API key missing"))
                    else:
                        ui.message(f"No responde {s.base_url}.\n¿Está arrancado el servidor y escucha en ese puerto?", C_BAD, "Sin conexión")
            elif key == "pick":
                self.pick_model()
            elif key == "mem":
                self.screen_memory()
            elif key == "test":
                self.check_server()
                ui.message(self.status[0] + (f"\nModelos: {', '.join(self.models_seen)}" if self.models_seen else ""), self.status[1], "Conexión")

    def pick_model(self):
        ui, s = self.ui, self.s
        self.check_server(fresh=5)            # the list from a moment ago is good enough: no wait
        if not self.models_seen:
            ui.message("No hay conexión: arranca un modelo local (opción A) o conecta a un servidor (opción B).", C_BAD)
            return
        items = [("(el primero que ofrezca el servidor)", "")] + [(m, "") for m in self.models_seen]
        i = ui.menu("Elegir modelo del servidor", items, sub=s.base_url)
        if i is not None:
            s.model = "" if i == 0 else self.models_seen[i - 1]
            config.save(s)

    # ---- ajustes ----
    def screen_config(self):
        state = {}
        s = self.s
        while True:
            fields = [
                Field("", "Ejecución", "section"),
                Field("temperature", "Temperatura", "float", s.temperature, hint="0 = determinista. Con >0 conviene usar varias muestras."),
                Field("workers", "Peticiones en paralelo", "int", s.workers, hint="No más que los slots (-np) del servidor."),
                Field("ld_max_new_tokens", T("Máx. tokens por respuesta", "Max tokens per answer"), "int", s.ld_max_new_tokens),
                Field("reasoning_budget", "Presupuesto de razonamiento", "int", s.reasoning_budget, hint="-1 = sin límite · 0 = sin razonar · N = tope de tokens de razonamiento por respuesta (llama-server: thinking_budget_tokens)"),
                Field("max_context", T("Contexto del servidor (tokens)", "Server context (tokens)"), "int", s.max_context),
                Field("timeout", "Timeout por petición (s)", "float", s.timeout),
                Field("extra_body", "JSON extra por petición", "text", s.extra_body,
                      hint='p. ej. {"chat_template_kwargs":{"enable_thinking":false}} para desactivar razonamiento en modelos que lo permiten'),
                Field("", "Interfaz", "section"),
                Field("ui_lang", "Idioma de la interfaz", "choice", s.ui_lang, [("en", "English"), ("es", "Español")], hint="También con la tecla L en el menú principal."),
                Field("splash", T("Pantalla de carga animada al abrir", "Animated loading screen at start"), "bool", s.splash),
                Field("", "Salida", "section"),
                Field("out_dir", "Carpeta de resultados", "text", s.out_dir),
                Field("leaderboard_auto", T("Añadir cada ejecución al leaderboard", "Add every run to the leaderboard"), "bool", s.leaderboard_auto),
                Field("", "El modelo y el servidor se eligen en «Modelo a evaluar».", "info", ""),
            ]
            kind, key = self.ui.form("Ajustes", fields, state, on_change=lambda k, fs: i18n.set_lang(next(f.value for f in fs if f.key == "ui_lang")) if k == "ui_lang" else None)
            sync(fields, s)
            i18n.set_lang(s.ui_lang)
            config.save(s)
            if kind == "back":
                return

    # ---- servidor local ----
    def screen_server(self):
        s = self.s
        state = {}
        found = servers.discover_models(s.model_dirs)
        while True:
            st = servers.managed_state()
            choices = [(m["path"], f"{m['name']}  ({servers.human(m['size'])})") for m in found]
            if s.srv_model and s.srv_model not in [c[0] for c in choices]:
                choices.insert(0, (s.srv_model, os.path.basename(s.srv_model)))
            binary = servers.find_binary()
            fields = [
                Field("", "Estado", "section"),
                Field("", "", "info", (f"● en marcha: PID {st['pid']} · puerto {st['port']} · alias {st['alias']}" if st else T("○ parado (de los lanzados desde aquí)", "○ stopped (servers started from here)")) +
                      (T(f"   · servidor: {binary}", f"   · server: {binary}") if binary else
                       T("   ⚠ no está llama-server ni `llama serve` en el PATH", "   ⚠ neither llama-server nor `llama serve` is on the PATH")), hint="" if st else ""),
                Field("", "Modelo", "section"),
                Field("srv_source", "Origen", "choice", s.srv_source, [("file", "Fichero GGUF local"), ("hf", "Hugging Face (-hf, se descarga)")]),
            ]
            if s.srv_source == "file":
                if choices:
                    cur = dict(choices).get(s.srv_model, os.path.basename(s.srv_model)) if s.srv_model else T("— ninguno: pulsa Enter para elegir —", "— none: press Enter to pick —")
                    fields.append(Field("pick_gguf", "Fichero GGUF", "action", cur,
                                        hint=T(f"{len(choices)} modelos encontrados · Enter para elegir", f"{len(choices)} models found · Enter to pick") + (f" · {s.srv_model}" if s.srv_model else "")))
                else:
                    searched = [str(r) for r in servers.model_roots(s.model_dirs) if r.is_dir()]
                    fields.append(Field("", T("No se encontró ningún .gguf. Añade una carpeta con «Añadir carpeta de modelos».",
                                              "No .gguf found. Add a folder with \"Add models folder\"."), "info", "",
                                        hint=T("Buscado en: ", "Searched: ") + (", ".join(searched) or T("(ninguna carpeta conocida existe)", "(none of the usual folders exist)"))))
            else:
                fields.append(Field("srv_hf", "Repositorio HF", "text", s.srv_hf, hint="usuario/repo[:quant], p. ej. LiquidAI/LFM3-1.7B-GGUF:Q4_K_M"))
            fields.append(Field("memline", T("Memoria (hf-mem): ", "Memory (hf-mem): "), "info", "…"))   # filled in by mem_tick, for the context (-c) set below
            fields += [
                Field("srv_lora", T("Adaptador LoRA (opcional)", "LoRA adapter (optional)"), "text", s.srv_lora,
                      hint=T("Un .gguf, un repo de HF (org/repo[:fichero.gguf]) o su URL, p. ej. harness-hub/LFM3-1.7B-opencode-GGUF. "
                             "Un repo PEFT usa su versión -GGUF. Se descarga al arrancar y se pasa con --lora.",
                             "A .gguf, a HF repo (org/repo[:file.gguf]) or its URL, e.g. harness-hub/LFM3-1.7B-opencode-GGUF. "
                             "A PEFT repo uses its -GGUF version. Downloaded at start and passed with --lora.")),
                Field("srv_lora_scale", T("Escala del LoRA", "LoRA scale"), "float", s.srv_lora_scale,
                      hint=T("1 = el adaptador tal cual (--lora); otro valor: --lora-scaled.", "1 = the adapter as is (--lora); any other value: --lora-scaled.")),
                Field("srv_alias", "Alias (nombre en la API)", "text", s.srv_alias,
                      hint=T("Vacío = nombre del fichero, o el del adaptador LoRA si hay uno (otra fila del leaderboard). Es el «model» que verá el bench.",
                             "Empty = the file's name, or the LoRA adapter's if there is one (another leaderboard row). It's the «model» the bench sees.")),
                Field("", T(f"Argumentos de {binary or 'llama-server'}", f"{binary or 'llama-server'} arguments"), "section"),
                Field("srv_port", "Puerto (--port)", "int", s.srv_port),
                Field("srv_ctx", "Contexto total (-c)", "int", s.srv_ctx, hint="Se reparte entre los slots (-np). Para las pruebas de 16k usa ≥ 20000 por slot."),
                Field("srv_parallel", "Slots (-np)", "int", s.srv_parallel),
                Field("srv_ngl", "Capas en GPU (-ngl)", "choice", s.srv_ngl, [("all", "all"), ("auto", "auto"), ("0", "0 (solo CPU)")]),
                Field("srv_fa", "Flash attention (-fa)", "choice", s.srv_fa, [("on", "on"), ("off", "off"), ("auto", "auto")]),
                Field("srv_jinja", "Plantilla Jinja (--jinja)", "bool", s.srv_jinja, hint="Necesaria para las pruebas de llamadas a herramientas."),
                Field("srv_reasoning", "Razonamiento (--reasoning)", "choice", s.srv_reasoning, [("default", "por defecto del modelo"), ("on", "on"), ("off", "off")],
                      hint="«off» desactiva el razonamiento (acelera, pero cambia qué mides)."),
                Field("srv_reasoning_budget", "Presupuesto de razonamiento (--reasoning-budget)", "int", s.srv_reasoning_budget,
                      hint="Tope de tokens de razonamiento para todo el servidor: -1 = sin límite, 0 = sin razonar."),
                Field("srv_extra", "Argumentos extra", "text", s.srv_extra, hint="Texto libre, p. ej. -ctk q8_0 -ctv q8_0 --temp 0.3"),
                Field("", "Acciones", "section"),
                Field("start", "Arrancar servidor", "action", ""),
                Field("stop", "Detener servidor", "action", ""),
                Field("log", "Ver log del servidor", "action", ""),
                Field("use", "Usar este servidor como modelo a evaluar", "action", ""),
                Field("adddir", "Añadir carpeta de modelos…", "action", ""),
                Field("rescan", "Buscar modelos de nuevo", "action", ""),
                Field("preview", "Ver el comando que se ejecutaría", "action", ""),
                Field("memdetail", T("Ver memoria estimada (hf-mem)", "Show estimated memory (hf-mem)"), "action", ""),
            ]

            def mem_tick(fs):
                from . import memory
                vals = {f.key: f.value for f in fs}
                src = vals.get("srv_source", s.srv_source)
                target = (vals.get("srv_hf") or "").strip() if src == "hf" else s.srv_model
                line = next((f for f in fs if f.key == "memline"), None)
                if not line:
                    return
                if src == "file" and target and not os.path.isfile(target):
                    line.value, line.hint = T("el fichero ya no existe: elige otro", "the file no longer exists: pick another"), "warn"
                    return
                if not target or not memory.command():
                    line.value = T("elige un modelo", "pick a model") if not target else T("hf-mem no está instalado", "hf-mem is not installed")
                    return
                ctx = vals.get("srv_ctx") if isinstance(vals.get("srv_ctx"), int) and vals.get("srv_ctx") > 0 else None
                r = memory.estimate_async(target, ctx=ctx)
                if r is None:
                    line.value = SPIN[int(time.time() * 8) % len(SPIN)] + T(" calculando…", " estimating…")
                    return
                line.value = memory.summary(r)
                f = memory.fit(memory.parts(r).get("total")) if not r.get("error") and not r.get("files") else ""
                if f:
                    line.value += "   " + {"ok": T("✔ cabe", "✔ fits"), "tight": T("⚠ justo", "⚠ tight"),
                                            "no": T("✗ no cabe en la memoria de este equipo", "✗ does not fit in this machine's memory")}[f]
                line.hint = "warn" if (r.get("error") or f in ("tight", "no")) else ""

            def on_change(k, fs):
                if k == "srv_source":                    # file <-> HF: rebuild so the GGUF picker / HF repo field appears at once
                    s.srv_source = next(f.value for f in fs if f.key == "srv_source")
                    state["rebuild"] = True

            kind, key = self.ui.form(T("Modelo local (GGUF)", "Local model (GGUF)") + f" · {binary or 'llama-server'}", fields, state,
                                     on_change=on_change, tick=mem_tick, sub="Elige el .gguf, ajusta los argumentos y pulsa «Arrancar servidor»: quedará como modelo a evaluar")
            sync(fields, s)
            if state.pop("rebuild", False):
                config.save(s)
                continue
            if s.srv_ngl not in ("all", "auto", "0") and not str(s.srv_ngl).isdigit():
                s.srv_ngl = "all"
            config.save(s)
            if kind == "back":
                return
            try:
                if key == "pick_gguf":
                    i = self.ui.menu("Elegir el modelo GGUF", [(c[1], c[0]) for c in choices],
                                     sel=max(0, [c[0] for c in choices].index(s.srv_model)) if s.srv_model in [c[0] for c in choices] else 0,
                                     sub="Ruta completa abajo · añade más carpetas con «Añadir carpeta de modelos…»")
                    if i is not None:
                        s.srv_model = choices[i][0]
                        s.srv_alias = ""
                        config.save(s)
                elif key == "start":
                    self.start_server()
                elif key == "stop":
                    self.ui.message("Servidor detenido." if servers.stop() else "No hay ningún servidor gestionado en marcha.", C_OK if st else C_WARN)
                elif key == "log":
                    if st:
                        self.ui.pager("Log de llama-server", servers.tail(st["log"]))
                    else:
                        self.ui.message("No hay servidor en marcha. El último log (si existe) está en ~/.config/slmbench/logs/.", C_WARN)
                elif key == "use":
                    s.base_url = f"http://127.0.0.1:{(st or {}).get('port', s.srv_port)}/v1"
                    s.model = (st or {}).get("alias", "") or s.srv_alias
                    config.save(s)
                    self.ui.message(f"Destino configurado:\n{s.base_url}\nmodelo: {s.model or '(el primero)'}", C_OK)
                elif key == "adddir":
                    d = self.ui.edit("Carpeta con modelos .gguf", "")
                    if d:
                        d = os.path.expanduser(d.strip())
                        if os.path.isdir(d):
                            if d not in s.model_dirs:
                                s.model_dirs.append(d)
                            config.save(s)
                            found = servers.discover_models(s.model_dirs)
                        else:
                            self.ui.message(f"No existe la carpeta: {d}", C_BAD)
                elif key == "rescan":
                    found = servers.discover_models(s.model_dirs)
                    self.ui.message(f"{len(found)} modelo(s) .gguf encontrados.", C_OK)
                elif key == "memdetail":
                    target = s.srv_hf.strip() if s.srv_source == "hf" else s.srv_model
                    if target:
                        self.screen_memory(target, ctx=s.srv_ctx if s.srv_ctx and s.srv_ctx > 0 else None)
                elif key == "preview":
                    b = servers.server_command() or ["llama-server"]
                    self.ui.message(" ".join(servers.build_command(s, b)), C_ACC, "Comando")
            except Exception as e:
                self.ui.message(str(e), C_BAD, "Error")

    def busy(self, title: str, text: str, fn):
        """Runs fn() in a thread with a spinner (the screen stays alive). Returns (result, error); Esc gives up waiting."""
        box: dict = {}

        def work():
            try:
                box["res"] = fn()
            except Exception as e:  # noqa: BLE001
                box["err"] = e
        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(0.15)                           # quick answers: no spinner flash at all
        ui, i, t0 = self.ui, 0, time.time()
        while th.is_alive():
            ui.scr.erase()
            ui.header(title)
            ui.put(3, 2, f"{SPIN[i % len(SPIN)]} {_(text)}", curses.color_pair(C_ACC) | curses.A_BOLD)
            ui.put(4, 4, f"{time.time() - t0:.0f} s", curses.color_pair(C_DIM))
            ui.footer(T("Esc: dejar de esperar", "Esc: stop waiting"))
            ui.scr.refresh()
            if ui.key(80) == KEY_ESC:
                return None, RuntimeError(T("cancelado", "cancelled"))
            i += 1
        return box.get("res"), box.get("err")

    def memory_target(self) -> tuple[str, int | None, tuple] | None:
        """(what to give hf-mem, context, fallbacks) for the model under test, or None if it can't be told: the GGUF / HF
        repo of the llama-server started from here, a .gguf path the server reports as its id, or a Hugging Face id. A
        plain id (safetensors repo, e.g. LiquidAI/LFM3-1.7B) goes to its GGUF repo first (LiquidAI/LFM3-1.7B-GGUF, with
        the quant llama-server reports or Q4_K_M), since GGUF is what slmbench serves; then the repo's quants; then the
        original repo."""
        s = self.s
        st = servers.managed_state()
        if st and f":{st.get('port')}" in s.base_url:
            cmd, target, ctx = st.get("cmd") or [], None, None
            for i, a in enumerate(cmd[:-1]):
                if a in ("-m", "--model", "-hf", "--hf-repo"):
                    target = cmd[i + 1]
                elif a in ("-c", "--ctx-size") and str(cmd[i + 1]).isdigit():
                    ctx = int(cmd[i + 1]) or None
            if target:
                return target, ctx, ()
        raw = self.current_model()
        if raw.lower().endswith(".gguf") and os.path.isfile(os.path.expanduser(raw)):
            return raw, s.max_context or None, ()
        mid = servers.clean_model_id(raw)
        if "/" not in mid or " " in mid:
            return None
        ctx = s.max_context or None
        if "gguf" in mid.lower():
            return mid, ctx, ()
        repo, _sep, tag = mid.partition(":")
        quant = (tag or (self.meta or {}).get("quant") or "Q4_K_M").upper()
        gguf = repo + "-GGUF"
        return f"{gguf}:{quant}", ctx, (gguf, repo)

    def screen_memory(self, target: str = "", ctx: int | None = None):
        """hf-mem: asks for a model (Hub id, id:QUANT or local path) and a context length, and shows the estimate."""
        from . import memory
        ui, s = self.ui, self.s
        if not memory.command():
            ui.message(T("hf-mem no está instalado: `pip install hf-mem` (o instala uv para usar `uvx hf-mem`).",
                         "hf-mem is not installed: `pip install hf-mem` (or install uv to use `uvx hf-mem`)."), C_BAD, "hf-mem")
            return
        if not target:
            guess = self.memory_target()
            target = ui.edit(T("Modelo (org/repo, org/repo:Q4_K_M o ruta a un .gguf)", "Model (org/repo, org/repo:Q4_K_M or a .gguf path)"),
                             guess[0] if guess else "")
            if not target or not target.strip():
                return
            c = ui.edit(T("Contexto en tokens (vacío = el máximo del modelo)", "Context in tokens (empty = the model's maximum)"),
                        str((guess[1] if guess else s.max_context) or ""))
            if c is None:
                return
            ctx = int(c) if c.strip().isdigit() else None
        self.memory_view(target.strip(), ctx)

    # ---- hf-mem: the memory view ----
    MEM_PARTS = (("weights", 209), ("kv", 205), ("overhead", 105))       # colour of each part in bars and legend
    MEM_CTX = [None, 2048, 4096, 8192, 16384, 32768, 65536, 131072]     # None = the model's maximum

    def memory_view(self, target: str, ctx: int | None = None, file: str | None = None, fallbacks: tuple = ()):
        """hf-mem's estimate, drawn: total and verdict, a stacked bar (weights · KV cache · overhead) over this machine's
        memory, a legend with the details, the context switchable with ←→. A repo with several GGUF files first lists
        them, each with its own bar; Enter opens one."""
        from . import memory
        ui = self.ui
        ram = memory.system_memory()
        name = os.path.basename(target) if os.path.exists(os.path.expanduser(target)) else target
        sel, top = 0, 0
        ctxs = list(self.MEM_CTX)
        if ctx not in ctxs:
            ctxs = sorted([c for c in ctxs if c] + [ctx]) if ctx else ctxs
            ctxs = [None] + [c for c in ctxs if c]
        shown = None                                   # the last estimate drawn: kept on screen while a new context loads
        while True:
            r = memory.estimate_async(target, ctx=ctx, file=file, fallbacks=fallbacks)
            if r and r.get("_target") and r["_target"] != target:      # a fallback answered: show what was measured
                target, fallbacks = r["_target"], ()
                name = target
                continue
            loading = r is None
            if loading and shown is not None and shown[0] == (target, file):
                r = shown[1]
            elif r is not None and not r.get("error"):
                shown = ((target, file), r)
            h, w = ui.size()
            ui.scr.erase()
            ui.header(T("Memoria del modelo", "Model memory") + " · hf-mem")
            W = max(40, min(w - 6, 104))
            x0 = 3
            ui.put(2, x0, name, curses.A_BOLD)
            if file:
                ui.put(2, x0 + len(name) + 1, f"· {file}", ui.color(ui.MUTED))
            if r is None:
                ui.put(4, x0, SPIN[int(time.time() * 8) % len(SPIN)] + "  " + T("Leyendo los metadatos (sin descargar los pesos)…",
                                                                                "Reading the metadata (without downloading the weights)…"),
                       curses.color_pair(C_ACC) | curses.A_BOLD)
                ui.progress(6, x0, W, (time.time() * 0.6) % 1.0)
            elif r.get("error"):
                ui.put(4, x0, "✗ " + T("hf-mem no pudo estimarlo", "hf-mem could not estimate it"), curses.color_pair(C_BAD) | curses.A_BOLD)
                lines = ui._wrap(r["error"], W - 2)[: h - 12]
                for i, l in enumerate(lines):
                    ui.put(6 + i, x0, l, ui.color(250))
                ui.put(7 + len(lines), x0, T("Prueba con org/repo, org/repo-GGUF:Q4_K_M o la ruta a un .gguf (m).",
                                    "Try org/repo, org/repo-GGUF:Q4_K_M or a path to a .gguf (m)."), ui.color(ui.MUTED) | _italic())
            elif r.get("files"):
                sel, top = self._memory_files(r, ram, x0, W, sel, top)
                cy_list = 8 + min(len(r["files"]), h - 12)
            cy = h - 4
            if r and r.get("files"):
                cy = min(h - 4, cy_list)
            if r and not r.get("error") and not r.get("files"):
                cy = min(h - 4, self._memory_single(r, ram, x0, W) + 1)
            # context selector (shared by both views)
            cur = T("máximo del modelo", "model's max") if ctx is None else f"{ctx:,} tokens"
            if r and not r.get("error") and ctx is None:
                mx = memory.parts(r["files"][0] if r.get("files") else r).get("ctx")
                cur += f" ({mx:,})" if mx else ""
            ui.put(cy, x0, T("Contexto", "Context") + "  ", ui.color(ui.MUTED, None, True))
            cx = x0 + len(T("Contexto", "Context")) + 2
            ui.put(cy, cx, "‹ ", ui.color(ui.MUTED))
            ui.put(cy, cx + 2, cur, curses.color_pair(C_ACC) | curses.A_BOLD)
            ui.put(cy, cx + 2 + len(cur), " ›", ui.color(ui.MUTED))
            if loading and r is not None:
                ui.put(cy, cx + 6 + len(cur), SPIN[int(time.time() * 8) % len(SPIN)] + T(" recalculando…", " recalculating…"),
                       curses.color_pair(C_ACC))
            listing = bool(r and r.get("files"))
            ui.footer(T("←→ contexto · " + ("↑↓ elegir · Enter detalle · " if listing else "") + "m otro modelo · t tabla · q volver",
                        "←→ context · " + ("↑↓ pick · Enter details · " if listing else "") + "m other model · t table · q back"))
            ui.scr.refresh()
            k = ui.key(120 if loading else -1)
            if k in ("q", "Q", KEY_ESC):
                return
            if k in (curses.KEY_LEFT, curses.KEY_RIGHT):
                i = ctxs.index(ctx) if ctx in ctxs else 0
                ctx = ctxs[(i + (1 if k == curses.KEY_RIGHT else -1)) % len(ctxs)]
            elif k in ("m", "M"):
                new = ui.edit(T("Modelo (org/repo, org/repo:Q4_K_M o ruta a un .gguf)", "Model (org/repo, org/repo:Q4_K_M or a .gguf path)"), target)
                if new and new.strip():
                    target, file, sel, top, fallbacks = new.strip(), None, 0, 0, ()
                    name = os.path.basename(target) if os.path.exists(os.path.expanduser(target)) else target
            elif k in ("t", "T") and r and not r.get("error"):
                ui.md_pager(T("Memoria estimada", "Estimated memory"), memory.markdown(r, name + (f" · {file}" if file else ""), ctx))
            elif listing and k in (curses.KEY_DOWN, "j"):
                sel = min(len(r["files"]) - 1, sel + 1)
            elif listing and k in (curses.KEY_UP, "k"):
                sel = max(0, sel - 1)
            elif listing and k in ("\n", "\r", curses.KEY_ENTER):
                self.memory_view(target, ctx, r["files"][sel].get("file"))

    def _memory_bar(self, y, x, width, p: dict, scale: float, ram: int):
        """Stacked bar: weights · KV cache · overhead, scaled so that `scale` bytes fill `width`; the rest a dim track,
        with a mark where this machine's memory ends when it falls inside the bar."""
        ui = self.ui
        col = 0
        for key, color in self.MEM_PARTS:
            n = int(round((p.get(key) or 0) / scale * width)) if scale else 0
            n = min(n, width - col)
            if n:
                ui.put(y, x + col, "█" * n, ui.color(color))
                col += n
        if col < width:
            ui.put(y, x + col, "░" * (width - col), ui.color(ui.RULE))
        if ram and ram < scale:                         # it doesn't fit: show where the memory runs out
            ui.put(y, x + int(ram / scale * width), "┃", ui.color(203, None, True))

    def _memory_single(self, r: dict, ram: int, x0: int, W: int):
        from . import memory
        ui = self.ui
        p = memory.parts(r)
        total = p["total"] or 0
        w_ = r.get("weights") or {}
        dt = w_.get("dtypes") or {}
        tb = sum((v.get("bytes") or 0) for v in dt.values()) or 1
        meta = [str(p["format"] or "").upper()]
        if p["params"]:
            meta.append(f"{p['params'] / 1e9:.2f}B params")
        meta += [f"{k} {v.get('bytes', 0) / tb * 100:.0f}%" if v.get('bytes', 0) / tb >= 0.01 else f"{k} <1%" for k, v in sorted(dt.items(), key=lambda kv: -(kv[1].get("bytes") or 0))[:4]]
        ui.put(3, x0, "  ·  ".join(m for m in meta if m), ui.color(ui.MUTED))
        # the total and the verdict
        fit = memory.fit(total, ram)
        big = f"≈ {memory.gb(total)}"
        ui.put(5, x0, big, ui.color(255, None, True))
        pill = {"ok": T(" ✔ CABE ", " ✔ FITS "), "tight": T(" ⚠ JUSTO ", " ⚠ TIGHT "), "no": T(" ✗ NO CABE ", " ✗ DOESN'T FIT ")}.get(fit, "")
        pc = {"ok": 78, "tight": 220, "no": 203}.get(fit, 245)
        if pill:
            ui.put(5, x0 + len(big) + 3, pill, ui.color(ui.INK, pc, True))
        if ram:
            ui.put(5, x0 + len(big) + 3 + len(pill) + 2, T(f"de {memory.gb(ram)} de este equipo · {total / ram * 100:.0f}%",
                                                           f"of this machine's {memory.gb(ram)} · {total / ram * 100:.0f}%"), ui.color(ui.MUTED))
        # what the total is made of (full width), then how much of this machine's memory it takes
        self._memory_bar(7, x0, W, p, total, 0)
        if ram:
            scale = max(ram, total)
            lab = T("memoria del equipo ", "this machine's memory ")
            ui.put(9, x0, lab, ui.color(ui.MUTED))
            bw = W - len(lab) - 10
            used = int(round(total / scale * bw))
            ui.put(9, x0 + len(lab), "▬" * used, ui.color(pc, None, True))
            ui.put(9, x0 + len(lab) + used, "▬" * max(0, bw - used), ui.color(ui.RULE))
            if ram < scale:
                ui.put(9, x0 + len(lab) + int(ram / scale * bw), "┃", ui.color(203, None, True))
            ui.put(9, x0 + len(lab) + bw + 2, memory.gb(ram), ui.color(ui.MUTED))
        # legend
        kv = r.get("kv_cache") or {}
        ov = r.get("overhead") or {}
        details = {
            "weights": T("Pesos", "Weights"),
            "kv": "KV cache",
            "overhead": "Overhead",
        }
        extra = {
            "weights": (f"{p['params'] / 1e9:.2f}B params" if p["params"] else "") + (f" · {p['file']}" if p["file"] else ""),
            "kv": (f"{kv.get('max_model_len'):,} ctx × {kv.get('batch_size') or 1} · {kv.get('dtype') or ''}" if kv else T("no generativo", "not generative"))
                  + (" · " + ", ".join(f"{n} {k}" for k, n in (kv.get("layers") or {}).items()) + T(" capas", " layers") if kv.get("layers") else ""),
            "overhead": " · ".join(f"{T(lab_es, lab_en)} {memory.gb(ov.get(k))}" for k, lab_es, lab_en in
                                   (("activations", "activaciones", "activations"), ("graphs", "grafos", "graphs"), ("runtime", "runtime", "runtime"),
                                    ("encoder", "encoder", "encoder")) if ov.get(k)),
        }
        y = 11
        for key, color in self.MEM_PARTS:
            v = p.get(key)
            if v is None:
                continue
            ui.put(y, x0, "■", ui.color(color, None, True))
            ui.put(y, x0 + 2, details[key], curses.A_BOLD)
            ui.put(y, x0 + 13, f"{memory.gb(v):>9}", ui.color(255, None, True))
            ui.put(y, x0 + 24, f"{v / total * 100:>3.0f}%" if total else "", ui.color(ui.MUTED))
            ui.put(y, x0 + 31, extra[key][: max(0, W - 31)], ui.color(250))
            y += 1
        if r.get("moe"):
            y += 1
            ui.put(y, x0, "MoE: " + ", ".join(f"{k} {v}" for k, v in r["moe"].items() if not isinstance(v, (dict, list)))[: W], ui.color(ui.MUTED))
        for wmsg in (r.get("warnings") or [])[:3]:
            y += 1
            ui.put(y, x0, ("⚠ " + wmsg)[: W], curses.color_pair(C_WARN))
        return y + 1

    def _memory_files(self, r: dict, ram: int, x0: int, W: int, sel: int, top: int) -> tuple[int, int]:
        """A repo with several GGUF files: one row per file, with its own stacked bar on a common scale."""
        from . import memory
        ui = self.ui
        h, w = ui.size()
        files = r["files"]
        sel = max(0, min(sel, len(files) - 1))
        tots = [f.get("total") or 0 for f in files]
        ui.put(3, x0, T(f"{len(files)} ficheros GGUF · de {memory.gb(min(tots))} a {memory.gb(max(tots))}",
                        f"{len(files)} GGUF files · from {memory.gb(min(tots))} to {memory.gb(max(tots))}")
               + (T(f" · este equipo: {memory.gb(ram)}", f" · this machine: {memory.gb(ram)}") if ram else ""), ui.color(ui.MUTED))
        lx = 0
        for key, color in self.MEM_PARTS:                 # legend of the colours
            lab = {"weights": T("pesos", "weights"), "kv": "KV cache", "overhead": "overhead"}[key]
            ui.put(4, x0 + lx, "■ ", ui.color(color, None, True))
            ui.put(4, x0 + lx + 2, lab, ui.color(250))
            lx += len(lab) + 5
        nw = min(34, max(len(f.get("file") or "") for f in files))
        bw = max(10, W - nw - 26)
        scale = max(max(tots), ram) if ram and ram < max(tots) * 1.5 else max(tots)
        ui.put(6, x0 + 2, T("FICHERO", "FILE").ljust(nw) + "   " + "TOTAL".rjust(8), ui.color(ui.MUTED, None, True))
        rows = h - 12
        if sel < top:
            top = sel
        if sel >= top + rows:
            top = sel - rows + 1
        for i, f in enumerate(files[top:top + rows]):
            idx, y = top + i, 7 + i
            cur = idx == sel
            bg = ui.SEL_BG if cur else None
            p = memory.parts(f)
            if cur:
                ui.sel_bar(y, x0 - 2, W + 4)
            fname = (f.get("file") or "?").removesuffix(".gguf")
            ui.put(y, x0 + 2, fname[:nw].ljust(nw), ui.color(255 if cur else 252, bg, cur))
            fit = memory.fit(p["total"], ram)
            ui.put(y, x0 + nw + 5, f"{memory.gb(p['total']):>8}", ui.color({"ok": 78, "tight": 220, "no": 203}.get(fit, 252), bg, True))
            self._memory_bar(y, x0 + nw + 15, bw, p, scale, ram)
        if len(files) > rows:
            ui.put(6, x0 + W - 8, f"{sel + 1}/{len(files)}", ui.color(ui.MUTED))
        return sel, top

    def start_server(self):
        ui, s = self.ui, self.s
        if (s.srv_lora or "").strip():                 # find and download the adapter first, with a spinner
            from . import lora
            info, err = self.busy(T("Adaptador LoRA", "LoRA adapter"), T(f"Buscando y descargando {s.srv_lora.strip()}…",
                                                                         f"Finding and downloading {s.srv_lora.strip()}…"),
                                  lambda: lora.resolve(s.srv_lora))
            if err:
                if str(err) != T("cancelado", "cancelled"):
                    ui.message(str(err), C_BAD, T("Adaptador LoRA", "LoRA adapter"))
                return
        st = servers.start(s)
        t0 = time.time()
        spin = "|/-\\"
        i = 0
        while True:
            if not servers.managed_state():
                ui.pager("El servidor se cerró al arrancar", servers.tail(st["log"]))
                return
            if servers.health(st["port"]):
                break
            ui.scr.erase()
            ui.header(T("Arrancando ", "Starting ") + (servers.find_binary() or "llama-server"))
            ui.put(3, 2, _(f"{spin[i % 4]} cargando el modelo… {time.time() - t0:.0f}s (Esc: seguir en segundo plano)"), curses.color_pair(C_WARN))
            tail = servers.tail(st["log"], 12).split("\n")
            for j, l in enumerate(tail[-(ui.size()[0] - 8):]):
                ui.put(5 + j, 2, l, curses.color_pair(C_DIM))
            ui.scr.refresh()
            k = ui.key(300)
            i += 1
            if k == KEY_ESC:
                return
            if time.time() - t0 > 600:
                ui.message("Tiempo de espera agotado (600 s). Revisa el log.", C_BAD)
                return
        s.base_url = f"http://127.0.0.1:{st['port']}/v1"
        s.model = st["alias"]
        config.save(s)
        ui.message(f"Servidor listo en {s.base_url}\nmodelo: {s.model}\n(ya es el destino del bench)", C_OK, "Servidor")

    # ---- tarea en segundo plano con log en vivo ----
    def live_task(self, title, worker, sub="", cancel_prompt="¿Cancelar? Se detendrá el proceso.", progress=False, background=False):
        """Ejecuta worker(on_line, cancel) en un hilo y muestra las últimas líneas. Devuelve {'err','res','cancelled','lines'}.
        background: m (or q/Esc → «back to the menu») leaves it running and returns at once with {'background': True}; the
        main menu tells when it ends (self.jobs)."""
        ui = self.ui
        lines: list[str] = []
        lock = threading.Lock()
        st = {"done": False, "err": None, "res": None, "cancelled": False, "lines": lines}
        cancel = threading.Event()

        def on_line(l):
            with lock:
                lines.append(l)
                if len(lines) > 3000:
                    del lines[:1000]

        def work():
            try:
                st["res"] = worker(on_line, cancel)
            except Exception as e:  # noqa: BLE001
                st["err"] = f"{type(e).__name__}: {e}"
            finally:
                st["done"] = True

        th = threading.Thread(target=work, daemon=True)
        t0 = time.time()
        th.start()
        spin = "|/-\\"
        tick = 0
        while True:
            h, w = ui.size()
            with lock:
                tail = lines[-(h - 7):] if h > 8 else []
            ui.scr.erase()
            ui.header(title, sub)
            el = time.time() - t0
            if st["done"]:
                ui.put(3, 2, _(f"Terminado en {int(el // 60)}m{int(el % 60):02d}s" + (f" · ERROR: {st['err']}" if st["err"] else "")),
                       curses.color_pair(C_BAD if st["err"] else C_OK) | curses.A_BOLD)
            else:
                ui.put(3, 2, _(f"{spin[tick % 4]} en curso… {int(el // 60)}m{int(el % 60):02d}s" + ("  (cancelando…)" if st["cancelled"] else "")),
                       curses.color_pair(C_WARN) | curses.A_BOLD)
            row0 = 5
            if progress and not st["done"]:
                act = [a for a in engine.active_runs() if a.get("started", 0) >= t0 - 5]
                if act:
                    a = act[0]
                    tot = a.get("total") or 0
                    bw = max(10, min(50, w - 30))
                    frac = min(1.0, a["done"] / tot) if tot else 0
                    eta = (el / a["done"] * (tot - a["done"])) if a["done"] and tot else 0
                    ui.progress(4, 2, bw, frac)
                    ui.put(4, 3 + bw, f" {a['done']}/{tot or '?'}"
                           + (f"  · ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta else ""), curses.color_pair(C_ACC) | curses.A_BOLD)
                    row0 = 6
            for i, l in enumerate(tail[: max(0, h - row0 - 2)]):
                ui.put(row0 + i, 2, l[: w - 4], curses.color_pair(C_DIM))
            agent_run = None
            if progress and not st["done"]:
                agent_run = next((a for a in engine.active_runs() if a.get("agent") and a.get("started", 0) >= t0 - 5), None)
            ui.footer(T("Cualquier tecla: continuar", "Any key: continue") if st["done"] else
                      (T("w ver el agente en vivo · ", "w watch the agent live · ") if agent_run else "")
                      + (T("m volver al menú (sigue en marcha) · q/Esc: opciones", "m back to the menu (keeps running) · q/Esc: options")
                         if background else T("q/Esc: cancelar", "q/Esc: cancel")))
            ui.scr.refresh()
            k = ui.key(250)
            tick += 1
            if st["done"]:
                if k is not None and k != curses.KEY_RESIZE:
                    return st
                continue
            if k in ("w", "W") and agent_run:
                self.watch_agent(agent_run["out"])
                continue
            if background and not st["cancelled"] and k in ("m", "M", "q", "Q", KEY_ESC):
                choice = 0 if k in ("m", "M") else ui.choose(title, [
                    (T("Volver al menú: la ejecución sigue en segundo plano", "Back to the menu: the run keeps going in the background"),
                     T("Síguela con «● en marcha» del pie o Ctrl+R; el menú avisa cuando termine.",
                       "Follow it with \"● running\" at the bottom or Ctrl+R; the menu tells you when it ends.")),
                    (T("Cancelar la ejecución", "Cancel the run"), _(cancel_prompt)),
                    (T("Seguir viéndola aquí", "Keep watching it here"), "")])
                if choice == 0:
                    st["thread"], st["cancel"], st["title"], st["background"] = th, cancel, title, True
                    self.jobs.append(st)
                    return st
                if choice == 1:
                    st["cancelled"] = True
                    cancel.set()
                continue
            if k in ("q", "Q", KEY_ESC) and not st["cancelled"]:
                if ui.confirm(cancel_prompt):
                    st["cancelled"] = True
                    cancel.set()

    # ---- ejecución en marcha (vista en vivo) ----
    def screen_running(self):
        """Progreso de las ejecuciones en marcha (de esta TUI o de `slmbench run` en otra terminal): barra, ETA,
        progreso por benchmark, últimas respuestas del modelo y log de lighteval. Se refresca cada segundo."""
        ui = self.ui
        sel, show_log, last, repaint = 0, False, None, True
        while True:
            runs = engine.active_runs()
            h, w = ui.size()
            if repaint:                         # cambio de ejecución o de panel: repintado completo (sin restos del anterior)
                ui.scr.clear()
                repaint = False
            else:
                ui.scr.erase()
            ui.header(T("Ejecución en marcha", "Running evaluation"),
                      T("se actualiza cada segundo", "refreshes every second") + (f" · {sel + 1}/{len(runs)}" if len(runs) > 1 else ""))
            if not runs:
                msg = T("No hay ninguna ejecución en marcha.", "No evaluation is running.")
                if last:
                    msg = T(f"La ejecución ha terminado. Informe: {last}/report.md", f"The run has finished. Report: {last}/report.md")
                ui.put(3, 2, msg, curses.color_pair(C_OK if last else C_DIM) | curses.A_BOLD)
                ui.footer(T("Cualquier tecla: volver", "Any key: back"))
                ui.scr.refresh()
                if ui.key() not in (None, curses.KEY_RESIZE):
                    return
                continue
            sel %= len(runs)
            a = runs[sel]
            last = a.get("out")
            rows = []
            try:
                with open(a["stats"]) as fh:
                    rows = [json.loads(l) for l in fh if l.strip()]
            except (OSError, ValueError):
                pass
            el = time.time() - a.get("started", time.time())
            tot, done = a.get("total") or 0, len(rows)
            eta = el / done * (tot - done) if done and tot and done < tot else 0
            agentic = bool(a.get("agent"))
            calls = rows
            if agentic:                         # rows = finished tasks; the model calls are in requests.jsonl
                try:
                    with open(Path(a["out"]) / "requests.jsonl") as fh:
                        calls = [json.loads(l) for l in fh if l.strip()]
                except (OSError, ValueError):
                    calls = []
            ok = [r for r in calls if "error" not in r]
            tps = [r["tps"] for r in ok if r.get("tps")]
            empty = sum(1 for r in ok if r.get("empty"))
            trunc = sum(1 for r in ok if r.get("finish") == "length")
            errs = len(calls) - len(ok)
            y = 2
            ui.put(y, 2, f"{T('Modelo', 'Model')}: ", curses.A_BOLD)
            ui.put(y, 12, f"{a.get('model', '?')}  ·  {a.get('base_url', '')}", curses.color_pair(C_ACC) | curses.A_BOLD)
            ui.put(y + 1, 2, f"{T('Etiqueta', 'Label')}: {a.get('label', '')}  ·  PID {a['pid']}"
                   + (f"  ·  {T('muestras', 'samples')}/bench: {a['max_samples']}" if a.get("max_samples") else ""), curses.color_pair(C_DIM))
            bw = max(10, min(60, w - 40))
            frac = min(1.0, done / tot) if tot else 0
            ui.progress(y + 3, 2, bw, frac)
            ui.put(y + 3, 3 + bw, f" {done}/{tot or '?'}  {int(frac * 100)}%", curses.color_pair(C_ACC) | curses.A_BOLD)
            ui.put(y + 4, 2, f"{T('Transcurrido', 'Elapsed')} {int(el // 60)}m{int(el % 60):02d}s"
                   + (f"  ·  ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta else "")
                   + (f"  ·  {sum(tps) / len(tps):.1f} tok/s" if tps else ""))
            warn = f"{T('vacías', 'empty')}: {empty}  ·  {T('truncadas', 'truncated')}: {trunc}  ·  {T('errores API', 'API errors')}: {errs}"
            if agentic:
                toks = sum(int(r.get("pt") or 0) + int(r.get("ct") or 0) for r in ok)
                from .agent import human
                warn = (f"{T('llamadas al modelo', 'model calls')}: {len(calls)}  ·  tokens: {human(toks)}  ·  {T('cortadas', 'cut')}: {trunc}  ·  "
                        f"{T('errores API', 'API errors')}: {errs}  ·  {T('resueltas', 'resolved')}: {sum(bool(r.get('resolved')) for r in rows)}/{done}")
            ui.put(y + 5, 2, warn, curses.color_pair(C_WARN if (empty or trunc or errs) else C_DIM))
            # progreso por benchmark
            y += 7
            ui.section(y, 2, T("Benchmarks", "Benchmarks"), min(w - 4, 100))
            per = a.get("per_task") or {sp: None for sp in a.get("specs", [])}
            counts: dict[str, int] = {}
            for r in rows:
                t = str(r.get("spec") or r.get("task") or "")
                for sp in per:
                    if sp.split("|")[0] in t.split("|"):
                        counts[sp] = counts.get(sp, 0) + 1
                        break
            names = {sp: (cat.catalog().get(cat.spec_to_key(sp) or "", {}).get("name") or sp.split("|")[0]) for sp in per}
            nw = max([len(v) for v in names.values()] + [12])
            for i, (sp, n) in enumerate(per.items()):
                if y + 1 + i >= h - 8:
                    break
                c = counts.get(sp, 0) if agentic or any(r.get("task") for r in rows) or not rows else "?"
                full = isinstance(c, int) and bool(n) and c >= n
                going = isinstance(c, int) and c and not full
                mark = "✔" if full else ("▸" if going else "·")
                style = curses.color_pair(C_OK) if full else (curses.color_pair(C_WARN) | curses.A_BOLD if going else curses.color_pair(C_DIM))
                yy = y + 1 + i
                ui.put(yy, 4, f"{mark} {names[sp]:<{nw}}", style)
                if isinstance(c, int) and n:
                    ui.progress(yy, 7 + nw, 20, c / n)
                ui.put(yy, 29 + nw, f"{c}/{n if n else '?'}", style)
                if agentic and isinstance(c, int) and c:
                    res = sum(bool(r.get("resolved")) for r in rows if r.get("spec") == sp)
                    ui.put(yy, 36 + nw, f"✔ {res} " + T("resueltas", "resolved"), curses.color_pair(C_OK if res else C_DIM))
            y += 2 + min(len(per), max(0, h - 10 - y))
            # últimas respuestas o log
            room = h - y - 2
            if show_log:
                ui.section(y, 2, T("Log de lighteval (l: últimas respuestas)", "lighteval log (l: latest answers)"), min(w - 4, 100))
                lines = []
                try:
                    with open(Path(a["out"]) / "run.log", errors="replace") as fh:
                        lines = fh.read().splitlines()[-max(1, room - 1):]
                except OSError:
                    lines = [T("(sin log: ejecución lanzada con una versión anterior)", "(no log: run started by an older version)")]
                for i, l in enumerate(lines[: max(0, room - 1)]):
                    ui.put(y + 1 + i, 2, l[: w - 4], curses.color_pair(C_DIM))
            elif agentic:                       # the agent's latest steps and finished tasks, newest first
                ui.section(y, 2, T("Últimos pasos del agente (l: log)", "Latest agent steps (l: log)"), min(w - 4, 100))
                try:
                    with open(Path(a["out"]) / "run.log", errors="replace") as fh:
                        lines = [l.rstrip() for l in fh.read().splitlines()[-400:]
                                 if " · #" in l or l.startswith(("✔", "✘", "→"))]
                except OSError:
                    lines = []
                for i, l in enumerate(list(reversed(lines))[: max(0, room - 1)]):
                    st = (curses.color_pair(C_OK) if l.startswith("✔") else curses.color_pair(C_BAD) if l.startswith("✘")
                          else curses.color_pair(C_ACC) if l.startswith("→") else curses.color_pair(C_WARN) if " ✂ " in l else 0)
                    ui.put(y + 1 + i, 2, l.strip()[: w - 4], st)
            else:
                ui.section(y, 2, T("Últimas respuestas del modelo (l: log)", "Latest model answers (l: log)"), min(w - 4, 100))
                for i, r in enumerate(list(reversed(rows))[: max(0, room - 1)]):
                    t = str(r.get("task") or "?").split("|")
                    tname = next((p for p in t if p and p != "custom" and not p.isdigit()), "?")
                    tname = cat.catalog().get(cat.spec_to_key(tname + "|0") or tname, {}).get("name") or tname
                    if "error" in r:
                        txt, st = f"{tname[:22]:<22} ERR   {r['error']}", curses.color_pair(C_BAD)
                    else:
                        # marcas ASCII: los símbolos (✓ ✂ ∅) tienen anchura ambigua y dejan restos en algunos terminales
                        flag = T("vacía", "empty") if r.get("empty") else (T("cortada", "cut") if r.get("finish") == "length" else "ok")
                        txt = f"{tname[:22]:<22} {flag:<7}{r.get('ct') or 0:>5} tok {r.get('t') or 0:>6.1f}s  {r.get('out', '')}"
                        st = curses.color_pair(C_WARN) if flag != "ok" else 0
                    ui.put(y + 1 + i, 2, txt[: w - 4], st)
            ui.footer((T("w agente en vivo · ", "w agent live · ") if agentic else "")
                      + T("l log/respuestas · ←→ otra ejecución · x cancelar · q volver", "l log/answers · ←→ other run · x cancel · q back"))
            ui.scr.refresh()
            k = ui.key(1000)
            if k in ("q", "Q", KEY_ESC):
                return
            repaint = k is not None
            if k in ("w", "W") and agentic:
                self.watch_agent(a["out"])
                continue
            if k in ("l", "L", "\t"):
                show_log = not show_log
            elif k in (curses.KEY_RIGHT, "n"):
                sel += 1
            elif k in (curses.KEY_LEFT, "p"):
                sel -= 1
            elif k in ("x", "X"):
                if ui.confirm(T(f"¿Cancelar la ejecución de {a.get('model')} (PID {a['pid']})?", f"Cancel the run of {a.get('model')} (PID {a['pid']})?")):
                    if a.get("cancel_file"):    # agent runs live in slmbench's own process: ask it to stop (it removes its sandboxes)
                        Path(a["cancel_file"]).touch()
                    else:
                        threading.Thread(target=engine.stop_process, args=(a["pid"],), daemon=True).start()

    # ---- leaderboard ----
    # brightness ramp of each mode's hue (xterm 256): a benchmark value goes from a dim tint to the full hue as it grows
    LB_HEAT = {209: (137, 173, 209), 75: (67, 68, 75), 205: (132, 169, 205), 141: (97, 98, 141), 78: (65, 71, 78), 43: (30, 37, 43), 214: (130, 172, 214)}
    LB_MEDAL = (220, 252, 173)                     # gold, silver, bronze

    @staticmethod
    def _lb_short(name: str) -> str:
        """Column title of a benchmark: drops the family prefix shared by the whole mode (MultiPL-E, BFCL, RULER NIAH)."""
        for pre in ("MultiPL-E ", "BFCL ", "RULER NIAH ", "SWE-bench Lite · "):
            name = name.replace(pre, "")
        return {"JavaScript": "JS", "LiveCodeBench": "LiveCode", "Parallel multiple": "Par. multi"}.get(name, name)

    def _lb_bar(self, y, x, width, v, hue, bg=None):
        """A thin meter: heavy line in the mode's hue (half-cell precision) over a light track."""
        ui = self.ui
        halves = int(round(max(0.0, min(1.0, v or 0)) * width * 2))
        full, half = divmod(halves, 2)
        ui.put(y, x, "━" * full + ("╸" if half else ""), ui.color(hue, bg, True))
        rest = width - full - half
        if rest > 0:
            ui.put(y, x + full + half, "─" * rest, ui.color(240 if bg else ui.RULE, bg))

    def _lb_heat(self, v, hue, bg=None):
        """Colour of a benchmark's value: grey below 25 %, then the mode's hue, brighter and bold as it grows."""
        ramp = self.LB_HEAT.get(hue, (245, 250, 255))
        if v is None or v < 0.25:
            return self.ui.color(242, bg)
        return self.ui.color(ramp[0], bg) if v < 0.5 else self.ui.color(ramp[1], bg) if v < 0.75 else self.ui.color(ramp[2], bg, True)

    def screen_leaderboard(self):
        """The leaderboard, drawn: one tab per mode (←→), a row per model + quant with its medal, score meter, speed and the
        score of each benchmark tinted with the mode's hue; Enter opens a row's card. The settings live behind «s»."""
        ui, s = self.ui, self.s
        lb = leaderboard
        lang = i18n.get_lang()
        mode_i, sel, top, col0 = 0, 0, 0, 0
        entries, stamp = None, None
        opened = None                                          # (when, ok) of the last «open in the browser»
        while True:
            d = lb.paths(s)["entries"]
            now = tuple(sorted((f.name, f.stat().st_mtime) for f in d.glob("*.json"))) if d.is_dir() else ()
            if now != stamp:                                   # a run that just finished adds or updates a row
                entries, stamp = lb.load_entries(s), now
            lang = i18n.get_lang()
            tabs = [(m, cat.MODES[m].get(lang, t), lb.ranked(entries, m)) for m, t in lb.modes()]
            tabs = [t for t in tabs if t[2]]
            if lb.has_harnesses(entries):                  # the agent mode seen from the agents' side
                tabs.append(("harness", "Harnesses", [h for h in lb.harness_table(entries) if h["models"]]))
            if lb.has_adapters(entries):                   # LoRA adapters against their base model
                tabs.append(("adapters", "Adapters", lb.adapter_rows(entries)))
            h, w = ui.size()
            ui.scr.erase()
            ui.header("Leaderboard")
            x0, W = 2, w - 4
            if not tabs:
                msg = [T("Todavía no hay modelos", "No models yet"),
                       T("Lanza benchmarks (menú 2): cada modelo y quant medido aparece aquí, con su medalla.",
                         "Run benchmarks (menu 2): every model and quant you measure shows up here, with its medal.")]
                ui.put(h // 2 - 2, max(2, (w - len(msg[0])) // 2), msg[0], ui.color(209, None, True))
                ui.put(h // 2, max(2, (w - len(msg[1])) // 2), msg[1], ui.color(ui.MUTED))
                ui.draw_bars(h // 2 - 4, max(2, (w - 24) // 2), 24)
                ui.footer(T("s ajustes · q volver", "s settings · q back"))
                ui.scr.refresh()
                k = ui.key(2000)
                if k in ("q", "Q", KEY_ESC):
                    return
                if k in ("s", "S"):
                    self.leaderboard_settings()
                continue
            mode_i %= len(tabs)
            mode, title, rows = tabs[mode_i]
            hue = self.MODE_COLOR.get(mode, 209)
            sel = max(0, min(sel, len(rows) - 1))
            # tabs: the current one is a pill in its hue, underlined; the rest a coloured dot
            x, under = x0, None
            for i, (m, t, rr) in enumerate(tabs):
                hh = self.MODE_COLOR.get(m, 209)
                label = f" {t} {len(rr)} "
                if i == mode_i:
                    ui.put(2, x, label, ui.color(ui.INK, hh, True))
                    under = (x, len(label))
                    x += len(label) + 2
                else:
                    ui.put(2, x, "●", ui.color(hh))
                    ui.put(2, x + 2, t, ui.color(250))
                    ui.put(2, x + 3 + len(t), str(len(rr)), ui.color(ui.MUTED))
                    x += len(t) + len(str(len(rr))) + 6
            n_models = len({e["name"] for e in entries})
            # the «open in the browser» button, right of the tabs (it turns green for a moment once opened)
            flash = opened and time.time() - opened[0] < 2.5
            if flash:
                label = " ✔ " + T("Abierto en el navegador", "Opened in the browser") + " " if opened[1] else " ✗ " + T("No se pudo abrir", "Could not open") + " "
                btn = [(label, ui.color(ui.INK, 78 if opened[1] else 203, True))]
            else:
                text = " " + T("Abrir en el navegador", "Open in the browser") + " ↗ "
                if x + len(text) + 5 > x0 + W:
                    text = " HTML ↗ "
                btn = [(" o ", ui.color(ui.INK, 209, True)), (text, ui.color(255, ui.SEL_BG, True))]
            bw_ = sum(len(t) for t, _a in btn)
            if x + bw_ + 1 <= x0 + W:
                bx = x0 + W - bw_
                for t, a in btn:
                    ui.put(2, bx, t, a)
                    bx += len(t)
            ui.put(3, x0, "─" * W, ui.color(ui.RULE))
            if under:
                ui.put(3, under[0], "━" * under[1], ui.color(hue, None, True))
            desc = (cat.MODES[mode].get(f"desc_{lang}") or cat.MODES[mode].get("desc_en", "")) if mode in cat.MODES else \
                T("Cada adaptador LoRA frente a su modelo base: ganancia en puntos sobre los benchmarks que ambos tienen medidos.",
                  "Each LoRA adapter against its base model: gain in points on the benchmarks both have measured.") if mode == "adapters" else \
                T("SWE-bench Lite con cada harness: el mismo modelo resuelve más o menos tareas según el agente que lo lleve.",
                  "SWE-bench Lite with each harness: the same model resolves more or fewer tasks depending on the agent that drives it.")
            right = T(f"{n_models} modelos · {len(entries)} filas", f"{n_models} models · {len(entries)} rows")
            ui.put(4, x0, desc[: max(0, W - len(right) - 3)], ui.color(250) | _italic())
            ui.put(4, x0 + W - len(right), right, ui.color(ui.MUTED))
            if mode in ("harness", "adapters"):
                if mode == "harness":
                    self._lb_harnesses(entries, rows, x0, W, h, hue)
                else:
                    self._lb_adapters(rows, x0, W, h, hue)
                ui.footer(T("←→ modo · o abrir HTML · s ajustes · q volver", "←→ mode · o open HTML · s settings · q back"))
                ui.scr.refresh()
                k = ui.key(500 if flash else 2000)
                if k in ("q", "Q", KEY_ESC):
                    return
                if k in (curses.KEY_RIGHT, "\t", "l"):
                    mode_i, sel, top, col0 = mode_i + 1, 0, 0, 0
                elif k in (curses.KEY_LEFT, curses.KEY_BTAB, "h"):
                    mode_i, sel, top, col0 = mode_i - 1, 0, 0, 0
                elif isinstance(k, str) and k.isdigit() and 1 <= int(k) <= len(tabs):
                    mode_i, sel, top, col0 = int(k) - 1, 0, 0, 0
                elif k in ("o", "O"):
                    try:
                        lb.open_html(lb.rebuild(s)["html"])
                        opened = (time.time(), True)
                    except Exception as ex:  # noqa: BLE001
                        opened = (time.time(), False)
                        ui.message(str(ex), C_BAD, "Error")
                elif k in ("s", "S"):
                    self.leaderboard_settings()
                    stamp = None
                continue
            # columns: rank · model · score + meter · tok/s · one per benchmark (the ones that don't fit scroll with < >)
            keys = [k for k in lb.mode_keys(mode) if any(k in e["benchmarks"] for e, _ in rows)]
            catl = cat.catalog()
            p5 = any(lb.mode_pass5(e, mode) is not None for e, _ in rows)
            nw = max(22, min(36, W // 4))
            xm, xs = x0 + 5, x0 + 5 + nw + 2
            mw = max(8, min(18, W // 9))                        # score meter
            xt = xs + 7 + mw
            tw = 9
            xb = xt + tw + 2
            cws = [max(7, min(11, len(self._lb_short(catl[k]["name"])) + 2)) for k in keys]
            col0 = max(0, min(col0, len(keys) - 1)) if keys else 0
            vis, xx = [], xb
            for j in range(col0, len(keys)):
                if xx + cws[j] > x0 + W - 2:                    # room for the «›» marker
                    break
                vis.append((j, xx))
                xx += cws[j] + 1
            hidden_l, hidden_r = col0, len(keys) - col0 - len(vis)
            hy = 6
            hd = ui.color(ui.MUTED, None, True)
            ui.put(hy, x0 + 1, "#", hd)
            ui.put(hy, xm, T("MODELO", "MODEL"), hd)
            ui.put(hy, xs, T("PUNTUACIÓN", "SCORE"), hd)
            ui.put(hy, xt, "tok/s".rjust(tw), hd)
            for j, cx in vis:
                ui.put(hy, cx, self._lb_short(catl[keys[j]]["name"])[: cws[j]].rjust(cws[j]), hd)
            if hidden_l:
                ui.put(hy, xb, "‹", ui.color(hue, None, True))
            if hidden_r:
                ui.put(hy, xx, "›", ui.color(hue, None, True))
            ui.put(hy + 1, x0, "─" * W, ui.color(ui.RULE))
            # the list: 2 lines per row and a blank one; a divider before the first row that isn't comparable
            blocks, y_of = [], {}
            for i, (e, sc) in enumerate(rows):
                if not sc["ranked"] and not any(b[0] == "div" for b in blocks):
                    blocks.append(("div", None, 2))
                blocks.append(("row", i, 3))
            off, pos = 0, []
            for b in blocks:
                pos.append(off)
                if b[0] == "row":
                    y_of[b[1]] = off
                off += b[2]
            y1, y2 = hy + 2, h - 3
            room = y2 - y1
            if y_of[sel] < top:
                top = y_of[sel] - (2 if sel and blocks[[b[1] for b in blocks].index(sel) - 1][0] == "div" else 0)
            elif y_of[sel] + 2 > top + room:
                top = y_of[sel] + 3 - room
            top = max(0, min(top, max(0, off - room)))
            for b, o in zip(blocks, pos):
                y = y1 + o - top
                if y < y1 - 2 or y >= y2:
                    continue
                if b[0] == "div":
                    lab = " " + T("No comparables aún: faltan benchmarks o se midió una muestra",
                                  "Not comparable yet: missing benchmarks or a sample run") + " "
                    if y >= y1:
                        ui.put(y, x0, "╌╌" + lab + "╌" * max(0, W - len(lab) - 2), ui.color(ui.MUTED))
                    continue
                i = b[1]
                e, sc = rows[i]
                m = e["meta"]
                cur = i == sel
                bg = ui.SEL_BG if cur else None
                if cur:
                    for yy in (y, y + 1):
                        if y1 <= yy < y2:
                            ui.put(yy, x0 - 1, " " * (W + 2), ui.color(255, bg))
                            ui.put(yy, x0 - 1, "▌", ui.color(hue, bg))
                if not (y1 <= y and y + 1 < y2):
                    continue
                if sc["ranked"] and i < 3:
                    ui.put(y, x0 + 1, f" {i + 1} ", ui.color(ui.INK, self.LB_MEDAL[i], True))
                else:
                    ui.put(y, x0 + 1, f"{i + 1:>2}", ui.color(ui.MUTED, bg))
                name = e["name"] if len(e["name"]) <= nw else "…" + e["name"][-(nw - 1):]
                ui.put(y, xm, name, ui.color(255 if sc["ranked"] else 250, bg, True))
                facts = [x for x in (lb.params_txt(m) if m.get("params_b") else "", m.get("quant", ""),
                                     f"{m['size_gb']} GB" if m.get("size_gb") else "", "LoRA" if m.get("lora") else "") if x]
                ft = " · ".join(facts)
                ui.put(y + 1, xm, ft[:nw], ui.color(ui.MUTED, bg))
                if not sc["ranked"]:
                    tag = (f"{sc['n']}/{sc['of']}" if not sc["complete"] else T("muestra", "sample"))
                    tx = xm + min(nw, len(ft)) + 2
                    if tx + len(tag) + 2 <= xs:
                        ui.put(y + 1, tx, f"⚠ {tag}", ui.color(220, bg))
                v = sc["score"]
                ui.put(y, xs, (lb.pct(v, 1).rstrip("%") if v is not None else "–").rjust(5),
                       ui.color(255 if sc["ranked"] else 250, bg, True))
                self._lb_bar(y, xs + 7, mw, v, hue if sc["ranked"] else 243, bg)
                if p5:
                    v5 = lb.mode_pass5(e, mode)
                    ui.put(y + 1, xs, (f"pass@5 {v5 * 100:.1f}" if v5 is not None else "")[: mw + 6], ui.color(ui.MUTED, bg))
                sp = lb.mode_speed(e, mode)
                ui.put(y, xt, f"{sp.get('gen_tps') or '–'}".rjust(tw), ui.color(250, bg))
                if mode == "long" and sp.get("prompt_tps"):
                    ui.put(y + 1, xt, f"pp {sp['prompt_tps']}".rjust(tw), ui.color(ui.MUTED, bg))
                for j, cx in vis:
                    r = e["benchmarks"].get(keys[j])
                    cw = cws[j]
                    if not r or r.get("value") is None:
                        ui.put(y, cx, "–".rjust(cw), ui.color(240, bg))
                        continue
                    val = f"{r['value'] * 100:.1f}" + ("" if r.get("full") else "*")
                    ui.put(y, cx, val.rjust(cw), self._lb_heat(r["value"], hue, bg))
                    self._lb_bar(y + 1, cx + 1, cw - 1, r["value"], hue, bg)
            more = []
            if hidden_l or hidden_r:
                more.append(T(f"‹ › más benchmarks ({hidden_l + hidden_r} ocultos)", f"‹ › more benchmarks ({hidden_l + hidden_r} hidden)"))
            if any(not (e["benchmarks"].get(k) or {}).get("full", True) for e, _ in rows for k in keys):
                more.append(T("* medido sobre una muestra", "* measured on a sample"))
            more.append(T("puntuación = media de los benchmarks del modo", "score = mean of the mode's benchmarks"))
            ui.hintline("  ·  ".join(more))
            ui.footer(T("←→ modo · ↑↓ fila · Enter ficha · <> columnas · o abrir HTML · d quitar · s ajustes · q volver",
                        "←→ mode · ↑↓ row · Enter card · <> columns · o open HTML · d remove · s settings · q back"))
            ui.scr.refresh()
            k = ui.key(500 if flash else 2000)
            if k in ("q", "Q", KEY_ESC):
                return
            if k in (curses.KEY_RIGHT, "\t", "l"):
                mode_i, sel, top, col0 = mode_i + 1, 0, 0, 0
            elif k in (curses.KEY_LEFT, curses.KEY_BTAB, "h"):
                mode_i, sel, top, col0 = mode_i - 1, 0, 0, 0
            elif isinstance(k, str) and k.isdigit() and 1 <= int(k) <= len(tabs):
                mode_i, sel, top, col0 = int(k) - 1, 0, 0, 0
            elif k in (curses.KEY_DOWN, "j"):
                sel = min(len(rows) - 1, sel + 1)
            elif k in (curses.KEY_UP, "k"):
                sel = max(0, sel - 1)
            elif k in (curses.KEY_NPAGE, " "):
                sel = min(len(rows) - 1, sel + max(1, room // 3))
            elif k == curses.KEY_PPAGE:
                sel = max(0, sel - max(1, room // 3))
            elif k in ("g", curses.KEY_HOME):
                sel = 0
            elif k in ("G", curses.KEY_END):
                sel = len(rows) - 1
            elif k in (">", ".") and hidden_r:
                col0 += 1
            elif k in ("<", ",") and hidden_l:
                col0 -= 1
            elif k in ("\n", "\r", curses.KEY_ENTER):
                e = rows[sel][0]
                ui.md_pager(f"Leaderboard · {e['name']}", "", head=lambda width, e=e: self.leaderboard_card(e, entries, width))
            elif k in ("o", "O"):
                try:
                    lb.open_html(lb.rebuild(s)["html"])
                    opened = (time.time(), True)
                except Exception as ex:  # noqa: BLE001
                    opened = (time.time(), False)
                    ui.message(str(ex), C_BAD, "Error")
            elif k in ("d", "D", curses.KEY_DC):
                e = rows[sel][0]
                label = f"{e['name']} {e['meta'].get('quant', '')}".strip()
                if ui.confirm(T(f"¿Quitar «{label}» del leaderboard? Se borran sus resultados de todos los modos.",
                                f"Remove '{label}' from the leaderboard? Its results in every mode are deleted.")):
                    lb.remove_entry(e["id"], s)
                    stamp = None
            elif k in ("s", "S"):
                self.leaderboard_settings()
                stamp = None

    def _lb_adapters(self, rows: list[dict], x0: int, W: int, h: int, hue: int) -> None:
        """The Adapters tab: a row per model served with a LoRA adapter (2 lines: the adapter and its base model), its mean
        gain over the base and, per mode, its score with the gain under it."""
        ui, lb = self.ui, leaderboard
        hd = ui.color(ui.MUTED, None, True)
        shown = [(m, t) for m, t in lb.modes() if any(m in r["modes"] for r in rows)]
        nw = max(24, min(40, W // 3))
        xg = x0 + 5 + nw + 2
        cw = max(10, min(14, (W - (xg - x0) - 10) // max(1, len(shown))))
        xm = xg + 10
        y = 6
        ui.put(y, x0 + 1, "#", hd)
        ui.put(y, x0 + 5, T("ADAPTADOR · BASE", "ADAPTER · BASE"), hd)
        ui.put(y, xg, T("GANANCIA", "GAIN").rjust(8), hd)
        for j, (m, t) in enumerate(shown):
            if xm + (j + 1) * cw > x0 + W:
                break
            ui.put(y, xm + j * cw, t[: cw - 1].rjust(cw - 1), ui.color(self.MODE_COLOR.get(m, ui.MUTED), None, True))
        ui.put(y + 1, x0, "─" * W, ui.color(ui.RULE))
        y += 2
        for i, r in enumerate(rows):
            if y + 1 >= h - 3:
                break
            e, base = r["entry"], r["base"]
            name = e["name"] if len(e["name"]) <= nw else "…" + e["name"][-(nw - 1):]
            ui.put(y, x0 + 1, f"{i + 1:>2}", ui.color(ui.MUTED))
            ui.put(y, x0 + 5, name, ui.color(255, None, True))
            bt = (("↳ " + base["name"]) if base else (("↳ " + r["base_name"] + T(" (sin medir)", " (not measured)")) if r["base_name"]
                                                   else "↳ " + T("base desconocida", "unknown base")))
            ui.put(y + 1, x0 + 5, bt[:nw], ui.color(ui.MUTED))
            g = r["gain"]
            ui.put(y, xg, lb.signed(g).rjust(8), ui.color(78 if (g or 0) > 0.0005 else 203 if (g or 0) < -0.0005 else 245, None, True))
            for j, (m, _t) in enumerate(shown):
                cx = xm + j * cw
                if cx + cw > x0 + W:
                    break
                x = r["modes"].get(m)
                if not x:
                    ui.put(y, cx, "–".rjust(cw - 1), ui.color(240))
                    continue
                ui.put(y, cx, (f"{x['score'] * 100:.1f}" + ("*" if x["partial"] else "")).rjust(cw - 1), self._lb_heat(x["score"], hue))
                if x["delta"] is not None:
                    d = x["delta"]
                    ui.put(y + 1, cx, lb.signed(d).rjust(cw - 1), ui.color(78 if d > 0.0005 else 203 if d < -0.0005 else 245))
            y += 3
        ui.hintline(T("ganancia = puntos sobre el modelo base, en los benchmarks que ambos tienen medidos · * medido sobre una muestra",
                      "gain = points over the base model, on the benchmarks both have measured · * measured on a sample"))

    def _lb_harnesses(self, entries: list[dict], hk: list[dict], x0: int, W: int, h: int, hue: int) -> None:
        """The Harnesses tab: a row per harness (version, models, mean resolved with a meter, best model, how the model acts)
        and the models × harnesses table, each model's best harness starred."""
        ui, lb = self.ui, leaderboard
        catl = cat.catalog()
        lang = i18n.get_lang()
        hd = ui.color(ui.MUTED, None, True)
        nw, mw = 16, max(8, min(18, W // 7))
        xv, xm, xr = x0 + 2 + nw, x0 + 10 + nw, x0 + 18 + nw
        xb = xr + 8 + mw
        y = 6
        ui.put(y, x0 + 2, "HARNESS", hd)
        ui.put(y, xv, T("VERS.", "VER."), hd)
        ui.put(y, xm, T("MODELOS", "MODELS"), hd)
        ui.put(y, xr, T("RESUELTAS (MEDIA)", "MEAN RESOLVED"), hd)
        ui.put(y, xb, T("MEJOR MODELO", "BEST MODEL"), hd)
        ui.put(y + 1, x0, "─" * W, ui.color(ui.RULE))
        y += 2
        for hh in sorted(hk, key=lambda r: -(r["mean"] or 0)):
            if y + 1 >= h - 3:
                break
            ui.put(y, x0 + 2, hh["name"][:nw - 1], ui.color(hue, None, True))
            ui.put(y, xv, (hh["version"] or "–")[:7], ui.color(250))
            ui.put(y, xm, str(hh["models"]).rjust(4), ui.color(250))
            v = hh["mean"]
            ui.put(y, xr, (f"{v * 100:.1f}" if v is not None else "–").rjust(5), ui.color(255, None, True))
            self._lb_bar(y, xr + 7, mw, v, hue)
            b = hh["best"]
            if b and b["value"]:
                txt = f"{b['name']} {b['quant']} · {b['value'] * 100:.1f}%" + ("" if b.get("full") else f" (n={b['n']})")
                ui.put(y, xb, txt[: max(0, x0 + W - xb)], ui.color(250))
            cost = (f"  ·  {lb._num(hh['turns'])} " + T("turnos", "turns") + f" · {lb._tok(hh['tokens'])} tokens "
                    + T("por tarea", "per task")) if hh.get("turns") is not None else ""
            ui.put(y + 1, x0 + 4, ((hh.get(f"note_{lang}") or hh["note_en"]) + cost)[: W - 6], ui.color(ui.MUTED) | _italic())
            y += 3
        rows = lb.harness_rows(entries)
        keys = [r["key"] for r in hk]
        if not rows or y + 3 >= h - 3:
            return
        y += 1
        cw = max(12, min(18, (W - 34) // max(1, len(keys) + 1)))
        ui.put(y, x0 + 2, T("MODELO", "MODEL"), hd)
        for j, k in enumerate(keys):
            ui.put(y, x0 + 30 + j * cw, catl[k].get("harness_name", k)[: cw - 1].rjust(cw - 1), hd)
        ui.put(y, x0 + 30 + len(keys) * cw, T("MEJOR", "BEST").rjust(cw - 1), hd)
        ui.put(y + 1, x0, "─" * W, ui.color(ui.RULE))
        y += 2
        for e in rows:
            if y >= h - 3:
                break
            bk = (lb.best_harness(e) or (None, 0))[0]
            name = f"{e['name']} {e['meta'].get('quant', '')}".strip()
            ui.put(y, x0 + 2, (name if len(name) <= 27 else "…" + name[-26:]), ui.color(255, None, True))
            for j, k in enumerate(keys):
                r = e["benchmarks"].get(k)
                cx = x0 + 30 + j * cw
                if not r or r.get("value") is None:
                    ui.put(y, cx, "–".rjust(cw - 1), ui.color(240))
                    continue
                val = ("★ " if k == bk else "") + f"{r['value'] * 100:.1f}" + ("" if r.get("full") else "*")
                ui.put(y, cx, val.rjust(cw - 1), self._lb_heat(r["value"], hue) | (curses.A_BOLD if k == bk else 0))
            ui.put(y, x0 + 30 + len(keys) * cw, (catl[bk].get("harness_name", bk) if bk else "–").rjust(cw - 1),
                   ui.color(hue if bk else ui.MUTED, None, True))
            y += 1
        ui.hintline(T("★ el mejor harness de cada modelo · * medido sobre una muestra de tareas · media = sobre los modelos medidos con ese harness",
                      "★ each model's best harness · * measured on a sample of the tasks · mean = over the models measured with that harness"))

    def leaderboard_card(self, e: dict, entries: list[dict], width: int) -> list:
        """A row of the leaderboard in detail: every mode's score and rank, each benchmark, speed, run settings and the other
        quantizations of the same model. Lines of (text, curses attr) for md_pager."""
        ui, lb = self.ui, leaderboard
        lang = i18n.get_lang()
        catl = cat.catalog()
        m = e["meta"]
        W = max(40, min(width, 110))
        dim, mut, bold = ui.color(ui.MUTED), ui.color(250), curses.A_BOLD

        def rule(label, hue=209):
            return [("── ", ui.color(hue)), (label, ui.color(hue, None, True)), (" " + "─" * max(0, W - len(label) - 4), ui.color(ui.RULE))]

        def meter(v, n, hue):
            halves = int(round(max(0.0, min(1.0, v or 0)) * n * 2))
            full, half = divmod(halves, 2)
            return [("━" * full + ("╸" if half else ""), ui.color(hue, None, True)), ("─" * (n - full - half), ui.color(ui.RULE))]

        facts = [x for x in (lb.params_txt(m) + " params" if m.get("params_b") else "", f"{m['size_gb']} GB" if m.get("size_gb") else "",
                             m.get("backend", ""), m.get("hardware", "")) if x]
        L = [[], [(e["name"], ui.color(255, None, True))] + ([("  ", 0), (f" {m['quant']} ", ui.color(ui.INK, 209, True))] if m.get("quant") else []),
             [(" · ".join(facts), dim)], []]
        mw = max(10, min(30, W - 44))
        L.append(rule(T("Puntuación por modo", "Score by mode")))
        L.append([])
        for mode, _t in lb.modes():
            rows = lb.ranked(entries, mode)
            sc = lb.mode_score(e, mode)
            if not sc["n"]:
                continue
            hue = self.MODE_COLOR.get(mode, 209)
            title = cat.MODES[mode].get(lang, mode)
            pos = next((i for i, (x, _) in enumerate(rows) if x["id"] == e["id"]), None)
            if sc["ranked"] and pos is not None:
                place = (f"#{pos + 1} " + T("de", "of") + f" {len(rows)}", ui.color(self.LB_MEDAL[pos] if pos < 3 else 250, None, pos < 3))
            else:
                place = ((f"⚠ {sc['n']}/{sc['of']} " + T("medidos", "measured")) if not sc["complete"] else "⚠ " + T("muestra", "sample"), ui.color(220))
            L.append([("● ", ui.color(hue)), (title[:19].ljust(19), bold), (lb.pct(sc["score"], 1).rjust(7), ui.color(255, None, True)), ("  ", 0)]
                     + meter(sc["score"], mw, hue if sc["ranked"] else 243) + [("  ", 0), place])
        for mode, _t in lb.modes():
            keys = [k for k in lb.mode_keys(mode) if k in e["benchmarks"]]
            if not keys:
                continue
            hue = self.MODE_COLOR.get(mode, 209)
            L += [[], rule(cat.MODES[mode].get(lang, mode), hue), []]
            for k in keys:
                r = e["benchmarks"][k]
                extra = []
                if r.get("pass5") is not None:
                    extra.append(f"pass@5 {r['pass5'] * 100:.1f}")
                extra.append(f"n={r.get('n')}" + ("" if r.get("full") else " · " + T("muestra", "sample")))
                if r.get("stderr"):
                    extra.append(f"± {r['stderr'] * 100:.1f}")
                L.append([("  " + catl.get(k, {}).get("name", k)[:20].ljust(21), mut), (lb.pct(r.get("value"), 1).rjust(7), self._lb_heat(r.get("value"), hue)),
                          ("  ", 0)] + meter(r.get("value"), mw, hue) + [("  " + " · ".join(extra), dim)])
            sp = lb.mode_speed(e, mode)
            if sp.get("gen_tps"):
                L.append([("  " + T("Velocidad", "Speed").ljust(21), mut), (f"{sp['gen_tps']}".rjust(7), ui.color(255, None, True)), (" tok/s", dim)]
                         + ([("  ·  prompt ", dim), (f"{sp['prompt_tps']} tok/s", mut)] if sp.get("prompt_tps") else [])
                         + ([(f"  ·  {sp['workers']} " + T("en paralelo", "in parallel"), dim)] if sp.get("workers") else []))
        b = list(e["benchmarks"].values())
        temps = sorted({x.get("temperature") for x in b if x.get("temperature") is not None})
        budgets = sorted({x.get("reasoning_budget", -1) for x in b})
        maxt = sorted({x.get("max_tokens") for x in b if x.get("max_tokens")})
        pk = sorted({f"{x['samples']}" + (f" @ T={x['sample_temperature']}" if x.get("sample_temperature") is not None else "")
                     for x in b if (x.get("samples") or 1) > 1})
        settings = [(T("Temperatura", "Temperature"), "/".join(map(str, temps)) or "–"),
                    (T("Presupuesto de razonamiento", "Reasoning budget"), "/".join(T("sin límite", "unlimited") if x < 0 else str(x) for x in budgets) or "–"),
                    (T("Tokens máximos", "Max tokens"), "/".join(map(str, maxt)) or "–"),
                    (T("Respuestas por problema (pass@k)", "Answers per problem (pass@k)"), ", ".join(pk) or "1"),
                    (T("Actualizado", "Updated"), e.get("updated") or "–"), ("id", e["id"])]
        L += [[], rule(T("Ajustes de la ejecución", "Run settings")), []]
        L += [[("  " + a.ljust(34), mut), (str(v_), 0)] for a, v_ in settings]
        sibs = sorted((x for x in entries if x["name"] == e["name"] and x["id"] != e["id"]), key=lambda x: -(x["meta"].get("size_gb") or 0))
        if sibs:
            L += [[], rule(T("Otras cuantizaciones", "Other quantizations")), []]
            for x in [e] + sibs:
                q = x["meta"].get("quant") or "?"
                segs = [("  " + ("▸ " if x is e else "  ") + q.ljust(9), ui.color(209, None, True) if x is e else bold),
                        (f"{x['meta']['size_gb']} GB".rjust(9) if x["meta"].get("size_gb") else " " * 9, dim)]
                for mode, _t in lb.modes():
                    sc = lb.mode_score(x, mode)
                    if sc["n"]:
                        segs += [("   ● ", ui.color(self.MODE_COLOR.get(mode, 209))), (lb.pct(sc["score"], 1) + ("" if sc["ranked"] else "*"), mut)]
                if x["speed"].get("gen_tps"):
                    segs.append((f"   {x['speed']['gen_tps']} tok/s", dim))
                L.append(segs)
        return L + [[]]

    def leaderboard_settings(self):
        ui, s = self.ui, self.s
        state = {}
        while True:
            entries = leaderboard.load_entries(s)
            p = leaderboard.paths(s)
            fields = [
                Field("", "Leaderboard", "section"),
                Field("", "", "info", f"{len(entries)} " + T("fila(s) (modelo + quant)", "row(s) (model + quant)") + f" · {_home_short(str(p['md']))}"),
                Field("leaderboard_auto", T("Añadir cada ejecución automáticamente", "Add every run automatically"), "bool", s.leaderboard_auto,
                      hint=T("Cada benchmark se guarda en la fila del modelo + quant: puedes medir un modo o un benchmark cada vez. Las canceladas no se "
                             "añaden; las parciales (muestras por benchmark) salen marcadas con n= y nunca sustituyen a un resultado completo.",
                             "Each benchmark is stored in the model + quant row, so you can measure one mode or benchmark at a time. Cancelled "
                             "runs are not added; partial ones (samples per benchmark) are marked n= and never replace a full result.")),
                Field("", "Acciones", "section"),
                Field("markdown", T("Ver leaderboard.md", "View leaderboard.md"), "action", ""),
                Field("rebuild", "Regenerar leaderboard.md / .html / .svg", "action", ""),
            ]
            kind, key = ui.form(T("Leaderboard · ajustes", "Leaderboard · settings"), fields, state)
            sync(fields, s)
            config.save(s)
            if kind == "back":
                return
            try:
                if key == "markdown":
                    ui.md_pager("leaderboard.md", leaderboard.markdown(leaderboard.load_entries(s)))
                elif key == "rebuild":
                    pp = leaderboard.rebuild(s)
                    ui.message(f"Regenerado:\n{pp['md']}\n{pp['html']}\n{pp['svg']}", C_OK)
            except Exception as e:  # noqa: BLE001
                ui.message(str(e), C_BAD, "Error")

    # ---- lanzar benchmarks ----
    def screen_run(self):
        ui, s = self.ui, self.s
        state = {}
        while True:
            catl = cat.catalog()
            mode_keys = cat.mode_benchmarks(s.mode)
            chosen = [k for k in (s.benchmarks or mode_keys) if k in mode_keys] or mode_keys
            fields = [
                Field("", T("Modo", "Mode"), "section"),
                Field("mode", T("Modo de evaluación", "Evaluation mode"), "choice", s.mode,
                      [(m, f"{info[i18n.get_lang()]} · {info['desc_' + i18n.get_lang()]}") for m, info in cat.MODES.items()] + [("all", T("Todo", "Everything"))],
                      hint=T("General · Programación (en el sandbox) · Herramientas (BFCL) · Contexto largo (RULER) · Agente (SWE-bench Lite) · "
                             "Todo (salvo el agente)",
                             "General · Coding (in the sandbox) · Tool calling (BFCL) · Long context (RULER) · Agent (SWE-bench Lite) · "
                             "Everything (but the agent)")),
                Field("", T("Benchmarks", "Benchmarks"), "section"),
            ]
            agentic = s.mode == "agent"
            if agentic:                                 # one benchmark per harness: picked in a pop-up
                miss = cat.missing_tools(mode_keys[0])
                names = ", ".join(catl[k]["harness_name"] for k in chosen) if len(chosen) < len(mode_keys) else T("todos", "all")
                fields.append(Field("harness", "Harness: " + names + "  ▸" + (f"  ⚠ {miss[0]}" if miss else ""), "action", "",
                                    hint=T("Enter: elegir el agente que usa el modelo (mini-SWE-agent, Pi, OpenCode o todos) · ? ficha",
                                           "Enter: pick the agent the model uses (mini-SWE-agent, Pi, OpenCode or all) · ? card")))
            for k in ([] if agentic else mode_keys):
                v = catl[k]
                miss = cat.missing_tools(k)
                label = f"{v['name']}" + (f"  ⚠ {', '.join(m.split(' (')[0] for m in miss)}" if miss else "")
                fields.append(Field(f"b:{k}", label, "bool", k in chosen,
                                    hint=(f"⚠ {T('falta', 'missing')}: {', '.join(miss)}  ·  " if miss else "")
                                    + f"{cat.describe(k, i18n.get_lang())}  ·  {v['n']:,} {T('tareas', 'tasks')} · `{v['metric']}` · ? = {T('ficha', 'card')}"))
            if agentic:
                fields += [
                    Field("", T("Opciones del agente", "Agent options"), "section"),
                    Field("ld_max_samples", T("Tareas (0 = todas las que tienen imagen)", "Tasks (0 = every task with an image)"), "int", s.ld_max_samples,
                          hint=T("Cada tarea tarda minutos (el agente da decenas de pasos y luego se pasan los tests): empieza con 5–20. "
                                 "Siempre las mismas y en el mismo orden, mezclando proyectos.",
                                 "Each task takes minutes (the agent takes dozens of steps, then the tests run): start with 5–20. "
                                 "Always the same ones in the same order, mixing projects.")),
                    Field("agent_max_tokens", T("Máx. tokens por respuesta", "Max tokens per answer"), "int", s.agent_max_tokens,
                          hint=T("Razonamiento incluido. Si se corta, el paso no trae comando y se pierde.", "Reasoning included. If cut, the step has no command and is wasted.")),
                    Field("reasoning_budget", T("Presupuesto de razonamiento", "Reasoning budget"), "int", s.reasoning_budget,
                          hint=T("-1 = sin límite · 0 = sin razonar · N = tope de tokens de razonamiento por respuesta", "-1 = unlimited · 0 = no reasoning · N = max reasoning tokens per answer")),
                    Field("temperature", T("Temperatura", "Temperature"), "float", s.temperature),
                    Field("agent_step_limit", T("Máx. pasos por tarea", "Max steps per task"), "int", s.agent_step_limit,
                          hint=T("Llamadas al modelo por tarea; al llegar, la tarea termina con lo que haya.", "Model calls per task; when reached, the task ends with what there is.")),
                    Field("agent_timeout", T("Tiempo por tarea (s)", "Time per task (s)"), "int", s.agent_timeout,
                          hint=T("Para el agente; la evaluación con los tests va aparte.", "For the agent; grading with the tests comes on top.")),
                    Field("agent_workers", T("Tareas en paralelo", "Parallel tasks"), "int", s.agent_workers,
                          hint=T("Como mucho 3. Cada una suma 4 GB de RAM al sandbox (y su imagen al disco); no más que los slots (-np) del servidor.",
                                 "At most 3. Each one adds 4 GB of RAM to the sandbox (and its image to the disk); no more than the server slots (-np).")),
                ]
            fields += ([] if agentic else [
                Field("ld_extra_tasks", T("Otras tareas de lighteval", "Other lighteval tasks"), "text", s.ld_extra_tasks,
                      hint=T("Separadas por comas, p. ej. aime24|0. Solo generativas.", "Comma-separated, e.g. aime24|0. Generative only.")),
                Field("", T("Opciones", "Options"), "section"),
                Field("ld_max_samples", T("Muestras por benchmark (0 = todas)", "Samples per benchmark (0 = all)"), "int", s.ld_max_samples,
                      hint=T("Con un SLM que razona, un benchmark completo puede tardar horas: empieza con 20.", "With a reasoning SLM a full benchmark can take hours: start with 20.")),
                Field("ld_max_new_tokens", T("Máx. tokens por respuesta", "Max tokens per answer"), "int", s.ld_max_new_tokens,
                      hint=T("Los modelos que razonan necesitan margen: si se queda corto, la respuesta sale vacía y puntúa 0.", "Reasoning models need room: if too low, the answer comes out empty and scores 0.")),
                Field("reasoning_budget", T("Presupuesto de razonamiento", "Reasoning budget"), "int", s.reasoning_budget,
                      hint=T("-1 = sin límite · 0 = sin razonar · N = tope de tokens de razonamiento por respuesta", "-1 = unlimited · 0 = no reasoning · N = max reasoning tokens per answer")),
                Field("temperature", T("Temperatura", "Temperature"), "float", s.temperature),
                Field("passk_n", T("pass@k: respuestas por problema (código, herramientas)", "pass@k: answers per problem (coding, tools)"), "int", s.passk_n,
                      hint=T("5 = pass@1 y pass@5 (5× más peticiones en programación y herramientas) · 1 = solo pass@1, más rápido. "
                             "Todas a la temperatura de arriba: a 0 las 5 salen casi iguales y pass@5 ≈ pass@1.",
                             "5 = pass@1 and pass@5 (5× more requests for coding and tool calling) · 1 = pass@1 only, faster. "
                             "All at the temperature above: at 0 the 5 come out nearly identical and pass@5 ≈ pass@1.")),
                Field("workers", T("Peticiones en paralelo", "Parallel requests"), "int", s.workers, hint=T("No más que los slots (-np) del servidor.", "No more than the server slots (-np).")),
            ]) + [
                Field("", T("Modelo a evaluar", "Model to evaluate"), "section"),
                Field("", "", "info", f"{self.current_model() or '—'} @ {s.base_url}"),
                Field("model", T("Cambiar el modelo a evaluar…", "Change the model to evaluate…"), "action", ""),
                Field("", "", "section"),
                Field("setup", T("Preparar el modo (comprobar herramientas y descargar datasets)", "Prepare the mode (check tools, download datasets)"), "action", ""),
                Field("go", "▶ " + T("Lanzar", "Run"), "action", ""),
            ]

            def pull(fs):
                for f in fs:
                    if f.key in ("ld_extra_tasks", "ld_max_samples", "ld_max_new_tokens", "reasoning_budget", "temperature", "workers",
                                 "passk_n", "agent_max_tokens", "agent_step_limit", "agent_timeout", "agent_workers"):
                        setattr(s, f.key, f.value)
                if agentic:                             # the harness is picked in its pop-up, not with checkboxes
                    return
                picked = [f.key[2:] for f in fs if f.key.startswith("b:") and f.value]
                s.benchmarks = [] if set(picked) == set(mode_keys) else picked

            def explain_key(f):
                if f.key.startswith("b:"):
                    self.show_card(f.key[2:])
                elif f.key == "harness":
                    self.show_card(chosen[0])

            def on_change(key, fs):
                if key == "mode":
                    s.mode = next(f.value for f in fs if f.key == "mode")
                    s.benchmarks = []
                    state["rebuild"] = True

            kind, key = ui.form(T("Lanzar benchmarks", "Run benchmarks"), fields, state, on_change=on_change, extra_keys={"?": explain_key},
                                sub=f"{T('Modelo', 'Model')}: {self.current_model() or '—'} @ {s.base_url}",
                                footer=T("↑↓ mover · Espacio marcar · ←→ cambiar modo · ? ficha del benchmark · q volver",
                                         "↑↓ move · Space toggle · ←→ change mode · ? benchmark card · q back"))
            if state.pop("rebuild", False):
                config.save(s)
                continue
            pull(fields)
            config.save(s)
            if kind == "back":
                return
            if key == "model":
                self.screen_model()
            elif key == "harness":
                self.pick_harness(mode_keys, chosen)
            elif key == "setup":
                r = self.live_task(T(f"Preparando el modo {s.mode}", f"Preparing mode {s.mode}"), lambda on_line, cancel: engine.setup(s.mode, on_line=on_line))
                if r["res"]:
                    miss = r["res"]["missing_tools"]
                    ui.message(T("Listo.", "Ready.") + ("\n" + T("Faltan herramientas: ", "Missing tools: ") + "; ".join(f"{k}: {', '.join(v)}" for k, v in miss.items()) if miss else ""),
                               C_WARN if miss else C_OK)
            elif key == "go":
                keys = [k for k in (s.benchmarks or mode_keys) if k in mode_keys]
                specs = engine.specs_for(keys, s.ld_extra_tasks)
                if not specs:
                    ui.message(T("Elige al menos un benchmark.", "Pick at least one benchmark."), C_BAD)
                    continue
                try:
                    engine.check_safe(specs)     # model code runs only in the sandbox
                except RuntimeError as e:
                    ui.message(str(e), C_BAD, "Sandbox")
                    continue
                self.check_server(fresh=5)
                if not self.models_seen:
                    ui.message(T(f"No hay conexión con {s.base_url}. Elige el modelo en «Modelo a evaluar».",
                                 f"No connection to {s.base_url}. Pick the model in \"Model to evaluate\"."), C_BAD)
                    continue
                if agentic:
                    why = cat.missing_tools(keys[0])
                    if why:
                        ui.message(T(f"El modo agente no está listo: {why[0]}. Usa «Preparar el modo».",
                                     f"Agent mode isn't ready: {why[0]}. Use \"Prepare the mode\"."), C_BAD, "Sandbox")
                        continue
                if any(k in ("lcb",) or catl[k].get("language") for k in keys) and not ui.confirm(
                        T("Los benchmarks de código ejecutan el código que genera el modelo dentro del sandbox (sbx). ¿Continuar?",
                          "Coding benchmarks run model-generated code inside the sandbox (sbx). Continue?"), default=True):
                    continue
                model = self.current_model()
                prog: dict = {}
                r = self.live_task(f"{T('Evaluando', 'Evaluating')} {model}", lambda on_line, cancel: engine.run(s, model, specs, on_line=on_line, cancel=cancel),
                                   sub=", ".join(keys), progress=True, background=True,
                                   cancel_prompt=T("¿Cancelar? Se guarda lo que haya terminado.", "Cancel? Finished results are kept."))
                if r.get("background"):                  # back to the main menu while it runs
                    return
                if r["err"]:
                    ui.message(f"Error: {r['err']}", C_BAD)
                    continue
                out = r["res"]
                meta = json.loads((Path(out) / "summary.json").read_text())["meta"]
                if meta.get("api_errors"):
                    ui.message(T(f"Peticiones fallidas: {meta['api_errors']}", f"Failed requests: {meta['api_errors']}") + "\n" + (meta.get("api_error_examples") or [""])[0][:200]
                               + "\n\n" + T("Detalle: Resultados → l (errores y log).", "Details: Results → l (errors and log)."), C_WARN)
                ui.md_pager(f"{T('Informe', 'Report')} · {Path(out).name}", (Path(out) / "report.md").read_text())

    def export_runs(self, runs: list[dict], sel: int, marked: set) -> None:
        """Pop-ups: which runs (marked / this one / every agent run / every run) and the format; then the dataset file."""
        from . import export
        ui, s = self.ui, self.s
        done = [r for r in runs if not r.get("incomplete")]
        agentic = [r for r in done if (Path(r["dir"]) / "agent").is_dir()]
        scopes = []
        if marked:
            scopes.append((T(f"Las {len(marked)} marcadas", f"The {len(marked)} marked runs"), [d for d in marked]))
        scopes.append((T("Esta ejecución: ", "This run: ") + Path(runs[sel]["dir"]).name[:40], [runs[sel]["dir"]]))
        if agentic:
            scopes.append((T(f"Todas las del modo agente ({len(agentic)})", f"Every agent-mode run ({len(agentic)})"), [r["dir"] for r in agentic]))
        scopes.append((T(f"Todas las ejecuciones terminadas ({len(done)})", f"Every finished run ({len(done)})"), [r["dir"] for r in done]))
        hint = T("Una fila por intento: la tarea, la sesión completa (razonamiento, respuestas, tool calls y salidas), la solución "
                 "(el parche en el modo agente) y si es correcta.",
                 "One row per attempt: the task, the whole session (reasoning, answers, tool calls and outputs), the solution "
                 "(the patch in the agent mode) and whether it's correct.")
        i = ui.choose(T("Exportar como dataset", "Export as a dataset"), [(l, hint) for l, _d in scopes])
        if i is None:
            return
        dirs = scopes[i][1]
        f = ui.choose(T("Formato", "Format"), [
            ("JSONL", T("Un objeto JSON por línea: fácil de leer y de procesar línea a línea.", "One JSON object per line: easy to read and stream.")),
            ("Parquet", T("Columnar y comprimido (zstd): para pandas, DuckDB o subirlo a Hugging Face.", "Columnar and compressed (zstd): for pandas, DuckDB or the Hugging Face Hub."))])
        if f is None:
            return
        fmt = ("jsonl", "parquet")[f]
        res, err = self.busy(T("Exportando", "Exporting"), T(f"Leyendo {len(dirs)} ejecuciones…", f"Reading {len(dirs)} runs…"),
                             lambda: export.export(dirs, fmt, None, s.out_dir, on_line=lambda _l: None))
        if err:
            if str(err) != T("cancelado", "cancelled"):
                ui.message(str(err), C_BAD)
            return
        path, n = res
        ui.message(T(f"✔ {n} filas → {path}", f"✔ {n} rows → {path}"), C_OK, T("Dataset exportado", "Dataset exported"))

    def pick_harness(self, mode_keys: list[str], chosen: list[str]) -> None:
        """Pop-up: which agent drives the model on SWE-bench Lite (one, or all of them on the same tasks)."""
        catl = cat.catalog()
        items = [(f"{catl[k]['harness_name']:<16} {catl[k]['harness_note_' + i18n.get_lang()]}", cat.describe(k, i18n.get_lang()))
                 for k in mode_keys]
        items.append((T("Todos", "All") + f"  ({len(mode_keys)}× " + T("tiempo", "time") + ")",
                      T("Los tres harnesses sobre las mismas tareas: compara el mismo modelo con un agente solo-bash y dos con tool calls.",
                        "The three harnesses on the same tasks: compares the same model with a bash-only agent and two tool-calling ones.")))
        sel = len(mode_keys) if len(chosen) == len(mode_keys) else mode_keys.index(chosen[0])
        i = self.ui.choose(T("Harness del agente", "Agent harness"), items, sel=sel)
        if i is None:
            return
        self.s.benchmarks = [] if i == len(mode_keys) else [mode_keys[i]]
        config.save(self.s)

    def show_card(self, key):
        from .benchmarks import explore
        ui = self.ui
        box = {"text": None}

        def work():                       # the sample task can take seconds the first time (lighteval, the Hub): show the card now
            try:
                box["text"] = explore.card(key, head=False)
            except Exception as e:  # noqa: BLE001
                box["text"] = f"Error: {e}"
        threading.Thread(target=work, daemon=True).start()

        def text():
            if box["text"] is not None:
                return box["text"]
            return "⋯ " + T("Cargando la tarea de muestra…", "Loading the sample task…")

        ui.md_pager(cat.catalog()[key]["name"], "", head=lambda w: self.card_view(key, text(), w),
                    live=lambda: (box["text"] is None, box["text"] is None))

    def card_view(self, key: str, sample_md: str, width: int) -> list:
        """The card: on a wide terminal the facts on the left and the sample task on the right; stacked otherwise."""
        if width < 128:
            return self.card_head(key, width) + mdview.render(sample_md, width)
        lw = max(52, min(66, int(width * 0.4)))
        rw = width - lw - 3
        hue = self.MODE_COLOR.get(cat.catalog()[key]["mode"], 209)
        left = self.card_head(key, lw, rule=False)
        right = [[], [(T("Tarea de muestra", "Sample task"), self.ui.color(hue, None, True)),
                      (T(" · lo que recibe el modelo", " · what the model receives"), curses.color_pair(C_DIM))],
                 [("─" * rw, curses.color_pair(C_DIM))]] + mdview.render(sample_md, rw)
        sep = ("│", curses.color_pair(C_DIM))

        def fit(segs, w):
            out, used = [], 0
            for text, style in segs:
                room = w - used
                if room <= 0:
                    break
                cut = ""
                for ch in text:
                    cw = mdview.dwidth(ch)
                    if used + cw > w:
                        break
                    cut += ch
                    used += cw
                if cut:
                    out.append((cut, style))
            return out + [(" " * (w - used), 0)]

        rows = max(len(left), len(right))
        return [fit(left[i] if i < len(left) else [], lw) + [(" ", 0), sep, (" ", 0)] + (right[i] if i < len(right) else [])
                for i in range(rows)]

    MODE_COLOR = {"general": 209, "code": 75, "tools": 205, "long": 141, "agent": 78, "harness": 43, "adapters": 214}    # the logo gradient, one hue per mode

    def card_head(self, key: str, width: int, rule: bool = True) -> list:
        """Styled header of a benchmark card: mode badge, name, a row of fact tiles, description, links."""
        import textwrap
        ui, v = self.ui, cat.catalog()[key]
        lang = i18n.get_lang()
        hue = self.MODE_COLOR.get(v["mode"], 209)
        acc, dim, bold = ui.color(hue, None, True), curses.color_pair(C_DIM), curses.A_BOLD
        badge = ui.color(17, hue, True)
        mode_name = cat.MODES.get(v["mode"], {}).get(lang, v["mode"])
        miss = cat.missing_tools(key)
        status = (("⚠ " + T("falta ", "missing ") + ", ".join(miss)), curses.color_pair(C_WARN) | bold) if miss else \
            ("✔ " + T("listo", "ready"), curses.color_pair(C_OK) | bold)
        tiles = [(f"{v['n']:,}", T("problemas", "problems"), acc),
                 (v["metric"] + (" · @5" if v.get("passk") else ""), T("métrica", "metric"), bold),
                 ("lighteval" if v.get("builtin") else T("tarea propia", "custom task"), T("origen", "origin"), bold),
                 (status[0], T("estado", "status"), status[1])]
        L = [[], [(f" {mode_name} ", badge), ("  " + key, dim)], [],
             [(v["name"], acc)], []]
        fit_n = max(1, (width - 1) // 15)
        per_row = len(tiles) if fit_n >= len(tiles) else (2 if fit_n >= 2 else 1)   # one row, or 2×2 in a narrow column
        cw = max(13, min(26, (width - 1) // per_row - 1))
        for r0 in range(0, len(tiles), per_row):
            chunk = tiles[r0:r0 + per_row]
            L.append([("╭" + "┬".join("─" * cw for _ in chunk) + "╮", dim)])
            row1, row2 = [("│", dim)], [("│", dim)]
            for big, small, attr in chunk:
                row1 += [(" " + big[:cw - 2].ljust(cw - 1), attr), ("│", dim)]
                row2 += [(" " + small[:cw - 2].ljust(cw - 1), dim), ("│", dim)]
            L += [row1, row2, [("╰" + "┴".join("─" * cw for _ in chunk) + "╯", dim)]]
        L.append([])
        for line in textwrap.wrap(v.get(f"desc_{lang}") or v.get("desc_es", ""), max(30, min(width, 96))):
            L.append([(line, 0)])
        L.append([])
        facts = [("Dataset", v["hf"], curses.color_pair(C_WARN)), ("lighteval", v["spec"], curses.color_pair(C_WARN))]
        if v.get("link"):
            facts.append((T("Referencia", "Reference"), v["link"], 0))
        if v.get("min_tokens"):
            facts.append((T("Mín. tokens por respuesta", "Min tokens per answer"), str(v["min_tokens"]), 0))
        for label, value, attr in facts:
            if len(label) + 2 + len(value) <= width:
                L.append([(label + "  ", dim), (value, attr)])
            else:                                                       # narrow column: value on its own line(s)
                L.append([(label, dim)])
                L += [[("  " + chunk, attr)] for chunk in textwrap.wrap(value, max(20, width - 2), break_on_hyphens=False)]
        if rule:
            L += [[], [("── " + T("Tarea de muestra · lo que recibe el modelo", "Sample task · what the model receives") + " ", acc),
                       ("─" * max(0, min(width, 96) - 50), dim)], []]
        return L

    # ---- explorar benchmarks ----
    # ---- chat with the model (llama cli) ----
    def screen_chat(self):
        """Interactive chat with a local GGUF through llama-cli / `llama cli`: pick the model, an optional system prompt,
        context and reasoning budget; the terminal is handed to llama cli and the TUI comes back when it exits."""
        ui, s = self.ui, self.s
        cli = servers.cli_command()
        if not cli:
            ui.message(T("No está llama-cli ni `llama cli` en el PATH: ", "Neither llama-cli nor `llama cli` is on the PATH: ")
                       + servers.INSTALL_HINT, C_BAD, "Chat")
            return
        found = servers.discover_models(s.model_dirs)
        default = s.chat_model or (s.srv_model if s.srv_source != "hf" else s.srv_hf) or (found[0]["path"] if found else "")
        choices = [(m["path"], f"{m['name']}  ({servers.human(m['size'])})") for m in found]
        if s.srv_source == "hf" and s.srv_hf and s.srv_hf not in [c[0] for c in choices]:
            choices.insert(0, (s.srv_hf, s.srv_hf + "  (Hugging Face)"))
        if default and default not in [c[0] for c in choices]:
            choices.insert(0, (default, os.path.basename(default) or default))
        if not choices:
            ui.message(T("No se encontró ningún .gguf. Añade carpetas en «Modelo a evaluar → A · Modelo local».",
                         "No .gguf found. Add folders in \"Model to evaluate → A · Local model\"."), C_WARN, "Chat")
            return
        state = {}
        while True:
            name = " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cli))
            fields = [
                Field("", T("Chatear con el modelo", "Chat with the model") + f"  ·  {name}", "section"),
                Field("chat_model", T("Modelo (GGUF)", "Model (GGUF)"), "choice", default if default in [c[0] for c in choices] else choices[0][0],
                      choices=choices, hint=T("Por defecto, el GGUF local que estás evaluando.", "Defaults to the local GGUF you are evaluating.")),
                Field("chat_system", T("Prompt de sistema (opcional)", "System prompt (optional)"), "text", s.chat_system),
                Field("chat_ctx", T("Contexto (-c)", "Context (-c)"), "int", s.chat_ctx),
                Field("chat_reasoning_budget", T("Presupuesto de razonamiento", "Reasoning budget"), "int", s.chat_reasoning_budget,
                      hint=T("-1 = sin límite · 0 = sin razonar", "-1 = unlimited · 0 = no reasoning")),
                Field("", "", "section"),
                Field("preview", T("Ver el comando", "Show the command"), "action", ""),
                Field("go", "▶ " + T("Empezar a chatear", "Start chatting"), "action", ""),
            ]
            kind, key = ui.form(T("Chat", "Chat"), fields, state,
                                sub=T("Dentro: escribe y pulsa Enter · /regen repite la respuesta · /clear borra el historial · /exit sale",
                                      "Inside: type and press Enter · /regen redo the answer · /clear reset the chat · /exit quit"))
            sync(fields, s)
            config.save(s)
            if kind == "back":
                return
            default = s.chat_model
            cmd = servers.chat_command(s, s.chat_model, s.chat_system, s.chat_ctx, s.chat_reasoning_budget)
            if key == "preview":
                ui.message(" ".join(cmd), C_ACC, T("Comando", "Command"))
            elif key == "go":
                self._run_in_terminal(cmd, T(f"Chat con {os.path.basename(s.chat_model) or s.chat_model}",
                                             f"Chat with {os.path.basename(s.chat_model) or s.chat_model}"))

    def screen_review(self, run_dir: str):
        """Failure explorer: benchmarks of a run -> its problems (failed ones first) -> prompt, answers and why they failed."""
        from . import review
        ui = self.ui
        tasks, err = self.busy(T("Respuestas", "Answers"), T("Leyendo las respuestas…", "Reading the answers…"),
                               lambda: review.load(run_dir))
        if err:
            if str(err) != T("cancelado", "cancelled"):
                ui.message(f"{type(err).__name__}: {err}", C_BAD)
            return
        if not tasks:
            ui.message(T("Esta ejecución no tiene detalles por problema.", "This run has no per-problem details."), C_WARN)
            return
        tsel = 0
        while True:
            items = [(f"{t['name']:<22} {len(t['samples']):>4} " + T("problemas", "problems") + "  ·  "
                      + f"{sum(x['status'] == 'fail' for x in t['samples'])} {T('mal', 'fail')}  ·  "
                      + f"{sum(x['status'] == 'partial' for x in t['samples'])} {T('parcial', 'partial')}  ·  "
                      + f"{sum(x['truncated'] for x in t['samples'])} {T('truncadas', 'truncated')}", "") for t in tasks]
            items.append((T("Resumen de toda la ejecución", "Summary of the whole run"), ""))
            i = ui.menu(T("Respuestas", "Answers") + f" · {Path(run_dir).name}", items, sel=tsel,
                        sub=T("Elige un benchmark para ver sus problemas", "Pick a benchmark to see its problems"))
            if i is None:
                return
            tsel = i
            if i == len(tasks):
                ui.md_pager(T("Resumen", "Summary"), review.summary_markdown(run_dir, tasks))
                continue
            self._review_task(tasks[i])

    def _review_task(self, task: dict):
        from . import review
        ui = self.ui
        only_bad, sel = True, 0
        order = {"fail": 0, "partial": 1, "n/a": 2, "pass": 3}
        while True:
            pool = sorted(task["samples"], key=lambda x: (order.get(x["status"], 9), x["index"]))
            shown = [x for x in pool if x["status"] != "pass"] if only_bad else pool
            items = [((T("▸ Mostrando solo los que fallan (Enter: ver todos)", "▸ Showing failures only (Enter: show all)") if only_bad
                       else T("▸ Mostrando todos (Enter: solo los que fallan)", "▸ Showing all (Enter: failures only)")), "")]
            items += [(_review_line(x), "") for x in shown]
            i = ui.menu(f"{task['name']} · {len(shown)}/{len(task['samples'])}", items, sel=sel,
                        sub=T("Enter: problema, respuestas y por qué fallan", "Enter: problem, answers and why they failed"))
            if i is None:
                return
            sel = i
            if i == 0:
                only_bad, sel = not only_bad, 0
                continue
            smp = shown[i - 1]
            md, err = self.busy(task["name"], T("Volviendo a comprobar cada respuesta (el código se ejecuta en el sandbox)…",
                                                "Re-checking every answer (code runs in the sandbox)…"),
                                lambda: review.sample_markdown(task, smp))
            if err:
                if str(err) != T("cancelado", "cancelled"):
                    ui.message(f"{type(err).__name__}: {err}", C_BAD)
                continue
            ui.md_pager(f"{task['name']} · #{smp['index']}", md)

    # ---- trayectorias de agentes (mini-SWE-agent, Pi, OpenCode) ----
    def screen_traces(self):
        """9 · Agent traces: the runs of the agent mode (newest first) -> their tasks -> every step. x exports them."""
        from . import trajectory as tr
        ui, s = self.ui, self.s
        sel = 0
        while True:
            runs, err = self.busy(T("Trazas del agente", "Agent traces"), T("Buscando ejecuciones del modo agente…", "Looking for agent-mode runs…"),
                                  lambda: [r for r in engine.list_runs(s.out_dir) if tr.has_trajectories(r["dir"])])
            if err:
                return
            if not runs:
                ui.message(T("Aún no hay ejecuciones del modo agente. Lánzalas en «2 · Lanzar benchmarks» con el modo Agente.",
                             "No agent-mode runs yet. Start one in \"2 · Run benchmarks\" with the Agent mode."), C_WARN, T("Trazas", "Traces"))
                return
            items = []
            for r in runs:
                heads = "  ".join(f"{(cat.catalog().get(cat.spec_to_key(t) or '', {}).get('harness_name') or t.split('|')[0])} {v * 100:.0f}%"
                                  for t, v in r["heads"].items())
                items.append((f"{r['date'][:16].replace('T', ' '):<16}  {_run_name(r)[:30]:<30}  "
                              + (heads or T("incompleta", "incomplete")), r["dir"]))
            k = ui.menu(T("Trazas del agente", "Agent traces"), items, sel=sel,
                        sub=T("Elige una ejecución: sus tareas y cada paso del agente", "Pick a run: its tasks and every step of the agent"),
                        footer=T("↑↓ mover · Enter abrir · q volver", "↑↓ move · Enter open · q back"))
            if k is None:
                return
            sel = k
            self.screen_trajectories(runs[k]["dir"])

    def watch_agent(self, run_dir: str):
        """The harness at work, live: the session of the task being solved (reasoning, answer, command or tool call, and
        what came back), from what slmbench's relay sees. Follows the newest task unless t picks another one."""
        from . import trajectory as tr
        ui = self.ui
        catl = cat.catalog()
        hs = {k: v.get("harness_name", k) for k, v in catl.items() if v.get("mode") == "agent"}
        state = {"rows": [], "size": -1, "chosen": None, "detailed": False}

        def refresh():
            try:
                size = (Path(run_dir) / "answers.jsonl").stat().st_size
            except OSError:
                size = 0
            if size != state["size"]:
                state["size"], state["rows"] = size, tr.live_rows(run_dir)

        def current():
            ts = tr.live_tasks(state["rows"])
            return state["chosen"] if state["chosen"] in ts else (ts[-1] if ts else None)

        def finished(cur) -> bool:
            done = tr._task_rows(Path(run_dir))
            return (catl.get(cur[0], {}).get("harness"), cur[1]) in done

        def running() -> bool:
            return any(a.get("out") == run_dir for a in engine.active_runs())

        def build(w):
            refresh()
            cur = current()
            if not cur:
                return mdview.render(T("# Agente en vivo\n\nEsperando la primera llamada al modelo…",
                                       "# Agent live\n\nWaiting for the first model call…"), w)
            done = finished(cur) or not running()
            if state["detailed"]:                   # e: the whole session, to scroll through
                title = hs.get(cur[0], cur[0]) + ("" if state["chosen"] else T(" · siguiendo la tarea más reciente", " · following the newest task"))
                return mdview.render(tr.live_markdown(state["rows"], cur[0], cur[1], title, done), w)
            return tr.live_lines(state["rows"], cur[0], cur[1], hs.get(cur[0], cur[0]), done, w, ui.size()[0])

        def live():
            refresh()
            go = running()
            return (state["size"], go), go

        while True:
            def title():
                cur = current()
                return T("Agente en vivo", "Agent live") + (f" · {cur[1]} [{hs.get(cur[0], cur[0])}]" if cur else "")

            k = ui.md_pager(title, "", head=build, live=live, follow=True, keys=("t", "T", "e", "E"),
                            hint=(T("e vista compacta · t otra tarea · G seguir", "e compact view · t other task · G follow")
                                  if state["detailed"] else
                                  T("e transcripción completa · t otra tarea", "e full transcript · t other task")))
            if k in ("e", "E"):
                state["detailed"] = not state["detailed"]
                continue
            if k not in ("t", "T"):
                return
            tasks = tr.live_tasks(state["rows"])
            if not tasks:
                continue
            items = [(T("▸ Seguir siempre la más reciente", "▸ Always follow the newest"), "")]
            items += [(f"{iid}  [{hs.get(key, key)}]", "") for key, iid in tasks]
            cur = current()
            i = ui.choose(T("Tarea", "Task"), items, sel=0 if state["chosen"] is None else 1 + tasks.index(cur))
            if i is not None:
                state["chosen"] = None if i == 0 else tasks[i - 1]

    def screen_trajectories(self, folder: str):
        """Agent runs: tasks (resolved or not, steps, cut answers) -> the task's steps -> one step: reasoning, response,
        bash command or tool call, and what came back. n/p move between steps without going back to the list."""
        from . import trajectory as tr
        ui = self.ui
        tasks, err = self.busy(T("Trayectorias", "Trajectories"), T("Leyendo las trayectorias…", "Reading the trajectories…"),
                               lambda: tr.find(folder))
        if err:
            if str(err) != T("cancelado", "cancelled"):
                ui.message(f"{type(err).__name__}: {err}", C_BAD)
            return
        if not tasks:
            ui.message(T("No hay trayectorias (*.traj.json, *.pi.jsonl) en esta carpeta.",
                         "No trajectories (*.traj.json, *.pi.jsonl) in this folder."), C_WARN)
            return
        graded = [t for t in tasks if t["resolved"] is not None]
        sub = (T(f"{sum(t['resolved'] for t in graded)}/{len(graded)} resueltas", f"{sum(t['resolved'] for t in graded)}/{len(graded)} resolved")
               if graded else T("sin evaluar", "not graded")) + " · " + T("Enter: pasos de la tarea", "Enter: the task's steps")
        sel = 0
        while True:
            i = ui.menu(T("Trayectorias", "Trajectories") + f" · {Path(folder).name}",
                        [(tr.task_line(t), f"{t['agent']} · {t['path']}") for t in tasks], sel=sel, sub=sub)
            if i is None:
                return
            sel = i
            self._trajectory_task(folder, tasks[i])

    def _trajectory_task(self, folder: str, task: dict):
        from . import trajectory as tr
        ui = self.ui
        try:
            t = tr.load_in(folder, task["path"])
        except (OSError, ValueError) as e:
            ui.message(f"{type(e).__name__}: {e}", C_BAD)
            return
        steps = t["steps"]
        head = [(T("▸ Resumen: evaluación, parche final y tarea", "▸ Summary: grading, final patch and task"), ""),
                (T("▸ Trayectoria completa (salidas largas recortadas)", "▸ Whole trajectory (long outputs clipped)"), "")]
        items = head + [(tr.step_line(s, last=s is steps[-1]), T("🧠 razonamiento · ✂ cortada por max_tokens · rc = código de salida del comando",
                                            "🧠 reasoning · ✂ cut by max_tokens · rc = the command's exit code")) for s in steps]
        sel = 0
        while True:
            i = ui.menu(f"{t['id']} · {len(steps)} " + T("pasos", "steps"), items, sel=sel,
                        sub=f"{t['agent']} · exit: {t['exit_status']} · " + T("Enter: ver el paso", "Enter: open the step"))
            if i is None:
                return
            sel = i
            if i == 0:
                ui.md_pager(f"{t['id']} · " + T("resumen", "summary"), tr.task_markdown(t, task.get("report")))
                continue
            if i == 1:
                ui.md_pager(f"{t['id']} · " + T("trayectoria", "trajectory"), tr.full_markdown(t, task.get("report")))
                continue
            n = i - len(head)
            while 0 <= n < len(steps):
                k = ui.md_pager(f"{t['id']} · " + T("paso", "step") + f" {n + 1}/{len(steps)}", tr.step_markdown(steps[n], last=n == len(steps) - 1),
                                keys=("n", "p", "N", "P"),
                                hint=T("n/p paso siguiente/anterior", "n/p next/previous step"))
                if k in ("n", "N"):
                    if n + 1 < len(steps):
                        n += 1
                elif k in ("p", "P"):
                    n = max(0, n - 1)
                else:
                    break
            sel = n + len(head) if 0 <= n < len(steps) else sel

    def _run_in_terminal(self, cmd: list[str], title: str):
        """Hands the terminal to an interactive program (curses suspended) and restores the TUI afterwards."""
        import signal as _signal
        import subprocess
        scr = self.ui.scr
        try:
            curses.mousemask(0)                         # otherwise clicks would reach the program as escape codes
        except curses.error:
            pass
        curses.def_prog_mode()
        curses.endwin()
        print(f"\n── slmbench · {title} ──\n$ {' '.join(cmd)}\n", flush=True)
        old = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)   # Ctrl+C belongs to llama cli, not to the TUI
        try:
            rc = subprocess.run(cmd, preexec_fn=lambda: _signal.signal(_signal.SIGINT, _signal.SIG_DFL)).returncode
            if rc not in (0, 130, -2):
                print(T(f"\n(terminó con código {rc})", f"\n(exited with code {rc})"))
            input(T("\nPulsa Enter para volver a slmbench…", "\nPress Enter to go back to slmbench…"))
        except (EOFError, OSError) as e:
            if isinstance(e, OSError):
                print(e)
        finally:
            _signal.signal(_signal.SIGINT, old)
            curses.reset_prog_mode()
            try:
                curses.mousemask(curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED | curses.BUTTON1_RELEASED)
            except curses.error:
                pass
            scr.clear()
            scr.refresh()

    def screen_benchmarks(self):
        sel = 0
        while True:
            catl = cat.catalog()
            keys, items = [], []
            for m, info in cat.MODES.items():
                for k, v in catl.items():
                    if v["mode"] == m:
                        keys.append(k)
                        miss = cat.missing_tools(k)
                        items.append((f"{info[i18n.get_lang()]:<13} {v['name']:<22} {v['n']:>6,}  {v['metric']:<24}" + (f"  ⚠ {', '.join(m.split(' (')[0] for m in miss)}" if miss else ""),
                                      (f"⚠ {', '.join(miss)}  ·  " if miss else "") + cat.describe(k, i18n.get_lang())))
            i = self.ui.menu(T("Benchmarks", "Benchmarks"), items, sel=sel,
                             sub=T("Enter: ficha con una tarea real de muestra (lo que recibe el modelo)", "Enter: card with a real sample task (what the model receives)"))
            if i is None:
                return
            sel = i
            self.show_card(keys[i])

    # ---- resultados ----
    def screen_results(self):
        ui, s = self.ui, self.s
        marked: set[str] = set()
        sel = 0
        runs, listed = [], 0.0
        while True:
            if time.time() - listed > 5:                # re-scan the results folder now and then, not on every frame
                runs, listed = engine.list_runs(s.out_dir), time.time()
            h, w = ui.size()
            ui.scr.erase()
            ui.header(T("Resultados", "Results"), s.out_dir)
            if not runs:
                ui.put(3, 2, T("Aún no hay ejecuciones guardadas.", "No saved runs yet."), curses.color_pair(C_WARN))
                ui.footer(T("q volver", "q back"))
                ui.scr.refresh()
                if ui.key() in ("q", "Q", KEY_ESC):
                    return
                continue
            sel = min(sel, len(runs) - 1)
            ui.put(2, 3, f"   {T('FECHA', 'DATE'):<16}  {T('MODELO', 'MODEL'):<30}  {'n':>5}  tok/s  {T('RESULTADOS', 'SCORES')}",
                   ui.color(ui.MUTED, None, True))
            top = max(0, sel - (h - 7) + 1)
            for i, r in enumerate(runs[top:top + h - 6]):
                idx = top + i
                y, cur = 3 + i, idx == sel
                bg = ui.SEL_BG if cur else None
                if cur:
                    ui.sel_bar(y, 1, w - 2)
                tag = (T("incompleta ", "incomplete ") if r.get("incomplete") else "") + (
                    T("repuntuada ", "rescored ") if r.get("kind") == "rescore" else T("reanudada ", "resumed ") if r.get("kind") == "resume" else "")
                mark = r["dir"] in marked
                ui.put(y, 3, "◉" if mark else "○", ui.color(78 if mark else ui.MUTED, bg, True))
                line = (f"{r['date'][:16].replace('T', ' '):<16}  {_run_name(r)[:30]:<30}  {str(r['max_samples'] or 'all'):>5}  "
                        f"{str(r['gen_tps'] or '-'):>5}  ")
                ui.put(y, 6, line, ui.color(255 if cur else 252, bg, cur))
                x = 6 + len(line)
                if tag:
                    ui.put(y, x, tag, ui.color(220, bg))
                    x += len(tag) + 1
                for t, v in r["heads"].items():             # each score coloured by how good it is
                    if x >= w - 2:
                        break
                    name = (cat.catalog().get(cat.spec_to_key(t) or '', {}).get('name') or t.split('|')[0])
                    ui.put(y, x, name + " ", ui.color(ui.MUTED, bg))
                    x += len(name) + 1
                    pct = f"{v * 100:.0f}%"
                    ui.put(y, x, pct, ui.color(78 if v >= 0.7 else 220 if v >= 0.4 else 203, bg, True))
                    x += len(pct) + 2
            ui.footer(T("↑↓ mover · Enter informe · e respuestas · l errores/log · r repuntuar · u reanudar · Espacio marcar · c comparar · x exportar · d borrar (marcadas) · q volver",
                        "↑↓ move · Enter report · e answers · l errors/log · r rescore · u resume · Space mark · c compare · x export · d delete (marked) · q back"))
            ui.scr.refresh()
            k = ui.key(1000)
            if k in (curses.KEY_DOWN, "j"):
                sel = min(len(runs) - 1, sel + 1)
            elif k in (curses.KEY_UP, "k"):
                sel = max(0, sel - 1)
            elif k in ("q", "Q", KEY_ESC):
                return
            elif k == " ":
                marked ^= {runs[sel]["dir"]}
            elif k in ("\n", "\r", curses.KEY_ENTER):
                rp = Path(runs[sel]["dir"]) / "report.md"
                if rp.exists():
                    ui.md_pager(f"{T('Informe', 'Report')} · {rp.parent.name}", rp.read_text())
                else:
                    ui.message(T("Ejecución incompleta: sin informe. Pulsa u para reanudarla.", "Incomplete run: no report. Press u to resume it."), C_WARN)
            elif k in ("e", "E"):
                from . import trajectory
                if trajectory.has_trajectories(runs[sel]["dir"]):
                    self.screen_trajectories(runs[sel]["dir"])
                else:
                    self.screen_review(runs[sel]["dir"])
            elif k in ("l", "L"):
                ui.md_pager(f"{T('Errores y log', 'Errors and log')} · {Path(runs[sel]['dir']).name}", engine.errors_markdown(runs[sel]["dir"]))
            elif k in ("r", "R", "u", "U"):
                d, again = runs[sel]["dir"], k in ("r", "R")
                fn = engine.rescore if again else engine.resume
                try:
                    engine.load_run(d)
                except RuntimeError as e:
                    ui.message(str(e), C_BAD)
                    continue
                if not again:
                    self.check_server(fresh=5)
                    if not self.models_seen:
                        ui.message(T(f"No hay conexión con {s.base_url}: arranca el servidor del modelo para reanudar.",
                                     f"No connection to {s.base_url}: start the model's server to resume."), C_BAD)
                        continue
                title = T("Repuntuando", "Rescoring") if again else T("Reanudando", "Resuming")
                res = self.live_task(f"{title} {Path(d).name}", lambda on_line, cancel: fn(d, s, on_line=on_line, cancel=cancel), progress=True,
                                     background=True)
                listed = 0.0
                if res.get("background"):
                    return
                if res["res"] and (Path(res["res"]) / "report.md").exists():
                    ui.md_pager(f"{T('Informe', 'Report')} · {Path(res['res']).name}", (Path(res["res"]) / "report.md").read_text())
                elif res["err"]:
                    ui.message(res["err"], C_BAD)
            elif k in ("c", "C"):
                dirs = [r["dir"] for r in runs if r["dir"] in marked]
                if len(dirs) < 2:
                    ui.message(T("Marca al menos dos ejecuciones con Espacio.", "Mark at least two runs with Space."), C_WARN)
                else:
                    ui.md_pager(T("Comparación", "Comparison"), engine.compare(dirs))
            elif k in ("x", "X"):
                self.export_runs(runs, sel, marked)
            elif k in ("d", "D"):
                dirs = [r["dir"] for r in runs if r["dir"] in marked] or [runs[sel]["dir"]]   # the marked runs, else the one under the cursor
                q = (T(f"¿Borrar {Path(dirs[0]).name}?", f"Delete {Path(dirs[0]).name}?") if len(dirs) == 1 else
                     T(f"¿Borrar las {len(dirs)} ejecuciones marcadas?", f"Delete the {len(dirs)} marked runs?"))
                if ui.confirm(q):
                    import shutil
                    for d in dirs:
                        shutil.rmtree(d, ignore_errors=True)
                    marked.difference_update(dirs)
                    listed = 0.0


def _review_line(smp: dict) -> str:
    v = smp["value"]
    pct = f"{v * 100:3.0f}%" if isinstance(v, (int, float)) and v == v else "  - "
    flags = (" " + T("truncada", "truncated") if smp["truncated"] else "") + (" " + T("vacía", "empty") if smp["empty"] else "")
    first = next((m.get("content", "") for m in smp["messages"] if m.get("role") == "user"), "")
    return f"#{smp['index']:<4} {smp['status']:<7} {pct}{flags:<12}  " + " ".join(str(first).split())[:70]


def _utf8_locale() -> bool:
    """curses only draws non-ASCII (borders, blocks, the wordmark) with a UTF-8 locale. WSL and minimal containers often start
    with LANG unset, i.e. the "C" locale: then try C.UTF-8 / en_US.UTF-8. Returns False if none is available."""
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
    if "utf" in (locale.nl_langinfo(locale.CODESET) or "").lower().replace("-", ""):
        return True
    for name in ("C.UTF-8", "C.utf8", "en_US.UTF-8", "en_US.utf8"):
        try:
            locale.setlocale(locale.LC_CTYPE, name)
            return True
        except locale.Error:
            continue
    return False


@contextlib.contextmanager
def _quiet_stderr():
    """While curses owns the terminal, what anything writes to stderr (huggingface_hub's download bars, a library's
    warning, from a background thread or a child process) goes to a log instead of over the screen. fd 2 is restored
    before an error's traceback is printed."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass
    try:
        config.config_dir().mkdir(parents=True, exist_ok=True)
        log = open(config.config_dir() / "tui-stderr.log", "ab", buffering=0)
        saved = os.dup(2)
    except OSError:
        yield
        return
    sys.stderr.flush()
    os.dup2(log.fileno(), 2)
    try:
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os.dup2(saved, 2)
        os.close(saved)
        log.close()


def run_trajectories(folder: str, settings=None):
    """`slmbench traj FOLDER`: the trajectory explorer on its own."""
    global ASCII_ONLY
    ASCII_ONLY = not _utf8_locale()
    s = settings or config.load()
    i18n.set_lang(s.ui_lang)
    with _quiet_stderr():
        curses.wrapper(lambda scr: App(scr, s).screen_trajectories(folder))


def run_tui(settings=None):
    global ASCII_ONLY
    ASCII_ONLY = not _utf8_locale()
    s = settings or config.load()
    i18n.set_lang(s.ui_lang)

    note = []

    def main(scr):
        app = App(scr, s)
        app.splash()
        app.main()
        note.append(getattr(app, "exit_note", ""))

    with _quiet_stderr():
        curses.wrapper(main)
    if note and note[0]:
        print(note[0])
