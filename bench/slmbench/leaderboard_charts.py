"""Comparison charts for the leaderboard page, in the spirit of Artificial Analysis: a ranked bar chart of each mode's score
and a score-vs-speed scatter. Plain inline SVG (no JS library); colors come from the page's CSS variables (--hue is the
mode's color), so they follow the light/dark theme. One series per chart, so no legend: the title names it."""
from __future__ import annotations

from html import escape


def _short(name: str, n: int = 18) -> str:
    base = name.split("/")[-1]
    return base if len(base) <= n else base[: n - 1] + "…"


def _label(e: dict) -> str:
    q = e["meta"].get("quant", "")
    return _short(e["name"]) + (f" {q}" if q else "")


def bars(rows: list[tuple[dict, dict]], title: str) -> str:
    """Score per model, best first. Rows that aren't comparable (missing benchmarks, sample runs) are drawn faded."""
    rows = [(e, sc) for e, sc in rows if sc["score"] is not None][:16]
    if not rows:
        return ""
    n = len(rows)
    W, top, bottom, left, right = 640, 34, 92, 40, 12
    H = 330
    ph = H - top - bottom
    slot = (W - left - right) / n
    bw = min(46, slot * 0.62)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}" class="chart">']
    for t in (0, .25, .5, .75, 1):                                     # recessive grid
        y = top + ph * (1 - t)
        o.append(f'<line x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{int(t * 100)}</text>')
    for i, (e, sc) in enumerate(rows):
        v = max(0.0, min(1.0, sc["score"]))
        x = left + slot * i + (slot - bw) / 2
        h = max(2.0, ph * v)
        y = top + ph - h
        r = min(4, h / 2)
        cls = "bar" if sc["ranked"] else "bar faded"
        note = "" if sc["ranked"] else (f" ({sc['n']} of {sc['of']} benchmarks)" if not sc["complete"] else " (sample run)")
        # rounded data-end, square at the baseline
        path = (f"M{x:.1f},{top + ph:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} "
                f"Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{top + ph:.1f} Z")
        o.append(f'<g class="hit"><title>{escape(e["name"])} {escape(e["meta"].get("quant", ""))}: {v * 100:.1f}{escape(note)}</title>'
                 f'<rect x="{left + slot * i:.1f}" y="{top}" width="{slot:.1f}" height="{ph}" class="hitbox"/>'
                 f'<path d="{path}" class="{cls}"/>'
                 f'<text x="{x + bw / 2:.1f}" y="{y - 7:.1f}" class="val" text-anchor="middle">{v * 100:.1f}</text>'
                 f'<text transform="translate({x + bw / 2:.1f},{top + ph + 12}) rotate(-38)" class="name" text-anchor="end">'
                 f'{escape(_label(e))}</text></g>')
    o.append(f'<line x1="{left}" x2="{W - right}" y1="{top + ph}" y2="{top + ph}" class="axis"/></svg>')
    return "".join(o)


def scatter(rows: list[tuple[dict, dict]], speed_of, title: str) -> str:
    """Score (y) against generation speed (x): top right is better and faster. Needs at least two models with a speed."""
    pts = []
    for e, sc in rows:
        tps = (speed_of(e) or {}).get("gen_tps")
        if sc["score"] is not None and tps:
            pts.append((e, sc, float(tps), max(0.0, min(1.0, sc["score"]))))
    if len(pts) < 2:
        return ""
    W, H, left, right, top, bottom = 640, 330, 46, 20, 30, 46
    pw, ph = W - left - right, H - top - bottom
    xmax = max(p[2] for p in pts) * 1.15
    step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if xmax / s <= 6)
    X = lambda v: left + pw * v / xmax                                  # noqa: E731
    Y = lambda v: top + ph * (1 - v)                                    # noqa: E731
    xs, ys = sorted(p[2] for p in pts), sorted(p[3] for p in pts)
    mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}" class="chart">']
    if len(pts) >= 3:                                                   # with 2 models the median corner is meaningless
        o += [f'<rect x="{X(mx):.1f}" y="{top}" width="{left + pw - X(mx):.1f}" height="{Y(my) - top:.1f}" class="quad"/>',
              f'<text x="{left + pw - 6}" y="{top + 14}" class="quadlabel" text-anchor="end">faster and better</text>']
    for t in (0, .25, .5, .75, 1):
        o.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{left - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{int(t * 100)}</text>')
    v = 0
    while v <= xmax:
        o.append(f'<text x="{X(v):.1f}" y="{top + ph + 18}" class="tick" text-anchor="middle">{v:g}</text>')
        v += step
    o.append(f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" class="axis"/>'
             f'<text x="{left + pw}" y="{H - 6}" class="axislabel" text-anchor="end">tokens per second →</text>'
             f'<text x="{left - 38}" y="{top - 16}" class="axislabel">score ↑</text>')
    placed: list[tuple[float, float, float, float]] = []                # label boxes already drawn
    dots = [(X(p[2]), Y(p[3])) for p in pts]

    def free(box):
        x0, y0, x1, y1 = box
        if x0 < left or x1 > left + pw or y0 < top - 4 or y1 > top + ph:
            return False
        if any(not (x1 < a[0] or x0 > a[2] or y1 < a[1] or y0 > a[3]) for a in placed):
            return False
        return not any(x0 - 6 < dx_ < x1 + 6 and y0 - 6 < dy_ < y1 + 6 for dx_, dy_ in dots)

    for e, sc, tps, s in sorted(pts, key=lambda p: -p[3]):
        x, y = X(tps), Y(s)
        cls = "dot" if sc["ranked"] else "dot faded"
        lab, text = "", _label(e)
        w = 6.2 * len(text)
        if len(placed) < 12:                                             # selective direct labels, never overlapping
            for dy in (0, -14, 14, -26, 26):
                for side in ((("start", 11), ("end", -11)) if x < left + pw * 0.7 else (("end", -11), ("start", 11))):
                    anchor, dx = side
                    x0 = x + dx if anchor == "start" else x + dx - w
                    box = (x0, y + dy - 9, x0 + w, y + dy + 4)
                    if free(box):
                        placed.append(box)
                        lab = f'<text x="{x + dx:.1f}" y="{y + dy + 4:.1f}" class="name" text-anchor="{anchor}">{escape(text)}</text>'
                        break
                if lab:
                    break
        o.append(f'<g class="hit"><title>{escape(e["name"])} {escape(e["meta"].get("quant", ""))}: score {s * 100:.1f}, '
                 f'{tps:g} tokens/s</title><circle cx="{x:.1f}" cy="{y:.1f}" r="14" class="hitbox"/>'
                 f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="{cls}"/>{lab}</g>')
    o.append("</svg>")
    return "".join(o)


CSS = """
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px;margin:0 0 18px}
.chartcard{margin:0;background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:16px 16px 8px}
.chartcard h3{font:600 16px/1.3 var(--display);margin:0 0 2px;letter-spacing:-.01em}
.chartcard p{margin:0 0 6px;color:var(--mut);font-size:13px}
svg.chart{display:block;width:100%;height:auto;overflow:visible}
svg.chart .grid{stroke:var(--line);stroke-width:1}
svg.chart .axis{stroke:var(--mut);stroke-width:1}
svg.chart .tick,svg.chart .axislabel{fill:var(--mut);font:500 11px var(--body)}
svg.chart .name{fill:var(--ink);font:500 11px var(--body)}
svg.chart .val{fill:var(--ink);font:700 12px var(--display)}
svg.chart .bar{fill:var(--hue)}
svg.chart .faded{fill:color-mix(in srgb,var(--hue) 30%,transparent);stroke:var(--hue);stroke-width:1.2;stroke-dasharray:3 3}
svg.chart .dot{fill:var(--hue);stroke:var(--paper);stroke-width:2}
svg.chart .dot.faded{fill:color-mix(in srgb,var(--hue) 35%,var(--paper));stroke:var(--hue);stroke-width:1.5;stroke-dasharray:none}
svg.chart .quad{fill:color-mix(in srgb,var(--hue) 9%,transparent)}
svg.chart .quadlabel{fill:var(--mut);font:600 11px var(--body)}
svg.chart .hitbox{fill:transparent}
svg.chart .hit:hover .bar,svg.chart .hit:hover .dot{filter:brightness(1.12)}
svg.chart .hit:hover .name{font-weight:700}
"""
