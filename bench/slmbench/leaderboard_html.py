"""The interactive leaderboard page (leaderboard/leaderboard.html): one tab per evaluation mode, sortable columns, light and
dark themes. Self-contained (inline CSS and JS); fonts come from Google Fonts and fall back to system fonts offline.

Visual language taken from the slmbench logo: navy ink and the yellow -> orange -> pink -> violet -> blue gradient. Each mode
takes one hue of that gradient, and a cell's tint grows with its score (a single hue, so the table reads as "more / less"
rather than as a red/green traffic light).
"""
from __future__ import annotations

import datetime as dt
from html import escape

# hue of each mode, taken from the logo gradient
MODE_HUE = {"general": "#ff8a3d", "code": "#3b82f6", "tools": "#ec4899", "long": "#8b5cf6", "agent": "#10b981"}
GRADIENT = ["#ffc93c", "#ffa83a", "#ff8a3d", "#ff6a5c", "#ff4f8b", "#d946ef", "#a855f7", "#8b5cf6", "#6366f1", "#3b82f6"]
BARS = [3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 22]


def _pct(v, d=1) -> str:
    return "–" if v is None else f"{v * 100:.{d}f}"


def _cell(v, sub: str = "", strong: bool = False) -> str:
    if v is None:
        return '<td class="num na" data-v="-1"><span aria-label="not measured">–</span></td>'
    s = f'<span class="sub">{sub}</span>' if sub else ""
    if strong:
        return (f'<td class="num score" data-v="{v:.4f}" style="--v:{v:.4f}"><span class="val">{_pct(v)}</span>'
                f'<span class="meter"></span>{s}</td>')
    return f'<td class="num" data-v="{v:.4f}"><span class="chip" style="--v:{v:.4f}">{_pct(v)}</span>{s}</td>'


def _speed(sp: dict, key: str) -> str:
    v = sp.get(key)
    return f'<td class="num plain" data-v="{v or 0}">{v if v else "–"}</td>'


def _name(name: str) -> str:
    """«org/model» with the organisation muted, so the model's own name stands out."""
    org, _, model = name.rpartition("/")
    return (f'<span class="org">{escape(org)}/</span>' if org else "") + escape(model)


def _facts(m: dict) -> list[str]:
    from . import leaderboard as lb
    return [x for x in (lb.params_txt(m) if m.get("params_b") else "", m.get("quant", ""),
                        f"{m['size_gb']} GB" if m.get("size_gb") else "") if x]


def _podium(rows, mode: str) -> str:
    """The three best comparable rows of the mode, as cards above the charts."""
    from . import leaderboard as lb
    top = [(e, sc) for e, sc in rows if sc["ranked"]][:3]
    if not top:
        return ""
    cards = []
    for i, (e, sc) in enumerate(top):
        medal = ("gold", "silver", "bronze")[i]
        sp = lb.mode_speed(e, mode).get("gen_tps")
        facts = "".join(f"<span>{escape(str(x))}</span>" for x in _facts(e["meta"]))
        cards.append(
            f'<article class="pod {medal}" style="--v:{sc["score"]:.4f}"><div class="podtop"><span class="medal {medal}">{i + 1}</span>'
            f'<span class="podplace">{("1st", "2nd", "3rd")[i]}</span></div>'
            f'<h3 class="podname" title="{escape(e["name"])}">{_name(e["name"])}</h3><div class="facts">{facts}</div>'
            f'<div class="podscore"><b>{_pct(sc["score"])}</b>' + (f'<span>{sp} tok/s</span>' if sp else "") +
            '</div><span class="meter"></span></article>')
    return f'<div class="podium">{"".join(cards)}</div>'


def _tip(rows, mode: str):
    """Stats of a model in this mode, for the tooltip of the charts: rank, score, pass@5, speed, facts and every benchmark."""
    from . import leaderboard as lb
    from .benchmarks import catalog as cat
    c = cat.catalog()
    by_id = {e["id"]: (i, sc) for i, (e, sc) in enumerate(rows)}

    def tip(e: dict) -> dict:
        i, sc = by_id[e["id"]]
        sp = lb.mode_speed(e, mode)
        status = "" if sc["ranked"] else (f"{sc['n']} of {sc['of']} benchmarks measured" if not sc["complete"]
                                          else "sample run: indicative only")
        bench = []
        for k in lb.mode_keys(mode):
            r = e["benchmarks"].get(k)
            if r and r.get("value") is not None:
                bench.append([c[k]["name"], round(r["value"], 4), None if r.get("full") else r.get("n"),
                              None if r.get("pass5") is None else round(r["pass5"], 4)])
        return {"name": e["name"], "quant": e["meta"].get("quant", ""), "rank": i + 1 if sc["ranked"] else None,
                "of": sum(1 for _, x in rows if x["ranked"]), "score": sc["score"], "p5": lb.mode_pass5(e, mode),
                "tps": sp.get("gen_tps"), "pp": sp.get("prompt_tps") if mode == "long" else None, "facts": _facts(e["meta"]), "status": status, "bench": bench}
    return tip


def _charts(rows, mode: str, title: str) -> str:
    from . import leaderboard as lb
    from .leaderboard_charts import bars, scatter
    tip = _tip(rows, mode)
    b = bars(rows, f"{title} score by model", tip)
    s = scatter(rows, lambda e: lb.mode_speed(e, mode), f"{title} score against speed", tip)
    cards = []
    if b:
        cards.append('<figure class="chartcard"><h3>Score by model</h3><p>Faded bars: not comparable yet (missing benchmarks '
                     f'or a sample run).</p>{b}</figure>')
    if s:
        cards.append('<figure class="chartcard"><h3>Score against speed</h3><p>Generation tokens per second on this machine. '
                     f'Top right is better and faster.</p>{s}</figure>')
    return f'<div class="charts">{"".join(cards)}</div>' if cards else ""


def page(entries: list[dict]) -> str:
    from . import leaderboard as lb
    from .benchmarks import catalog as cat
    c = cat.catalog()
    tabs, panels = [], []
    for mode, title in lb.modes():
        rows = lb.ranked(entries, mode)
        if not rows:
            continue
        keys = [k for k in lb.mode_keys(mode) if any(k in e["benchmarks"] for e, _ in rows)]
        p5 = any(lb.mode_pass5(e, mode) is not None for e, _ in rows)
        long_ = mode == "long"
        heads = [("#", "rank"), ("Model", "model"), ("Score", "num")] + ([("Score at 5", "num")] if p5 else []) \
            + [(c[k]["name"], "num") for k in keys] + [("Tokens/s", "num")] + ([("Prompt tokens/s", "num")] if long_ else [])
        th = "".join(f'<th class="{cls}" scope="col"><button type="button">{escape(h)}</button></th>' for h, cls in heads)
        body, divided = [], False
        n_ranked = sum(1 for _, sc in rows if sc["ranked"])
        for i, (e, sc) in enumerate(rows):
            m = e["meta"]
            if not sc["ranked"] and not divided:
                divided = True
                note = ("Not comparable yet: some benchmarks of the mode are missing, or the run used a sample. "
                        "Their score averages a different set.")
                body.append(f'<tr class="divider"><td colspan="{len(heads)}"><span>{note}</span></td></tr>')
            medal = ["gold", "silver", "bronze"][i] if sc["ranked"] and i < 3 else ""
            rank = f'<span class="medal {medal}">{i + 1}</span>' if medal else f'<span class="rk">{i + 1}</span>'
            tag = "" if sc["ranked"] else (f'<span class="tag">{sc["n"]} of {sc["of"]} measured</span>' if not sc["complete"]
                                           else '<span class="tag">sample run</span>')
            hw = f' title="{escape(m["hardware"])}"' if m.get("hardware") else ""
            model = (f'<td class="model" data-v="{escape(e["name"].lower())}"><span class="name"{hw}>{_name(e["name"])}</span>{tag}'
                     f'<span class="facts">{"".join(f"<span>{escape(str(x))}</span>" for x in _facts(m))}</span></td>')
            tds = [f'<td class="rank" data-v="{i}">{rank}</td>', model, _cell(sc["score"], strong=True)]
            if p5:
                tds.append(_cell(lb.mode_pass5(e, mode)))
            for k in keys:
                r = e["benchmarks"].get(k)
                if not r:
                    tds.append(_cell(None))
                    continue
                sub = " ".join(x for x in ((f"at 5: {_pct(r['pass5'])}" if r.get("pass5") is not None else ""),
                                           ("" if r.get("full") else f"n={r['n']}")) if x)
                tds.append(_cell(r["value"], sub))
            sp = lb.mode_speed(e, mode)
            tds.append(_speed(sp, "gen_tps"))
            if long_:
                tds.append(_speed(sp, "prompt_tps"))
            body.append(f'<tr class="{"ranked" if sc["ranked"] else "unranked"}">' + "".join(tds) + "</tr>")
        info = cat.MODES[mode]
        how = ("Scores are pass@1. With several answers per problem, pass@5 is shown under each score and averaged in "
               "“Score at 5”." if p5 else "")
        hue = MODE_HUE.get(mode, "#ff8a3d")
        tabs.append(f'<button type="button" role="tab" id="tab-{mode}" aria-controls="p-{mode}" data-mode="{mode}" '
                    f'style="--hue:{hue}"><span class="dot"></span>{escape(title)}<span class="count">{len(rows)}</span></button>')
        lead = (f"{n_ranked} of {len(rows)} rows have every benchmark measured on the full set." if n_ranked != len(rows)
                else f"All {len(rows)} rows have every benchmark measured on the full set.")
        panels.append(
            f'<section class="panel" role="tabpanel" id="p-{mode}" aria-labelledby="tab-{mode}" style="--hue:{hue}">'
            f'<div class="intro"><h2>{escape(title)}</h2><p>{escape(info.get("desc_en", ""))} {lead} {how}</p></div>'
            + _podium(rows, mode) + _charts(rows, mode, title) +
            f'<div class="scroll card"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>')

    details = []
    for e in sorted(entries, key=lambda x: x["name"].lower()):
        m, b = e["meta"], e["benchmarks"].values()
        temps = sorted({x.get("temperature") for x in b if x.get("temperature") is not None})
        budgets = sorted({x.get("reasoning_budget", -1) for x in b})
        maxt = sorted({x.get("max_tokens") for x in b if x.get("max_tokens")})
        pk = sorted({f"{x['samples']}" + (f" at T={x['sample_temperature']}" if x.get("sample_temperature") is not None else "")
                     if (x.get("samples") or 1) > 1 else "1"
                     for x in b if "samples" in x})
        cells = [e["name"], m.get("quant", "–"), m.get("hardware", "–"), m.get("backend", "–"),
                 "/".join(map(str, temps)) or "–", ", ".join(pk) or "–",
                 "/".join("unlimited" if x < 0 else str(x) for x in budgets) or "–", "/".join(map(str, maxt)) or "–", e.get("updated") or "–"]
        details.append("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in cells) + "</tr>")
    det_head = ["Model", "Quant", "Hardware", "Backend", "Temperature", "pass@k answers per problem", "Reasoning budget", "Max tokens", "Updated"]

    bars = "".join(f'<i style="height:{h * 4}px;background:{GRADIENT[int(j * (len(GRADIENT) - 1) / (len(BARS) - 1))]}"></i>'
                   for j, h in enumerate(BARS))
    today = dt.date.today().isoformat()
    n_models = len({e["name"] for e in entries})
    n_bench = len({k for e in entries for k in e["benchmarks"]})
    if entries:
        main = (f'<nav class="tabs" role="tablist" aria-label="Evaluation mode">{"".join(tabs)}</nav>{"".join(panels)}'
                f'<details class="settings"><summary>Run settings for each row</summary><div class="scroll card"><table class="plain">'
                f'<thead><tr>{"".join(f"<th scope=col>{h}</th>" for h in det_head)}</tr></thead><tbody>{"".join(details)}</tbody>'
                f'</table></div></details>')
    else:
        main = ('<div class="empty"><h2>No models yet</h2><p>Run <code>slmbench run</code> and every benchmark you measure will '
                'show up here, one row per model and quantization.</p></div>')
    stat = lambda n, one, many: f'<li><b>{n}</b>{one if n == 1 else many}</li>'      # noqa: E731
    stats = (stat(n_models, "model", "models") + stat(len(entries), "row", "rows") + stat(n_bench, "benchmark", "benchmarks")
             if entries else "") + f'<li><b>{today}</b>updated</li>'
    from .leaderboard_charts import CSS as CHART_CSS
    return TEMPLATE.replace("/*CHARTS*/", CHART_CSS).replace("%BARS%", bars).replace("%STATS%", stats).replace("%MAIN%", main)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLM Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f6f7fb;--paper:#ffffff;--ink:#1c2240;--mut:#636a8c;--line:#e4e7f1;--soft:#eef0f7;--glow:.55;--focus:#3b82f6;
--shadow:0 1px 2px rgba(28,34,64,.05),0 10px 30px -14px rgba(28,34,64,.16);
--grad:linear-gradient(90deg,#ffc93c,#ff8a3d 22%,#ff4f8b 45%,#a855f7 70%,#3b82f6);
--display:"Lexend",ui-rounded,"SF Pro Rounded",system-ui,sans-serif;--body:"Manrope",system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121530;--paper:#191d3b;--ink:#eceefa;--mut:#9aa0c3;--line:#2a2f55;--soft:#20254a;--glow:.32;--focus:#7aa7ff;
  --shadow:0 1px 2px rgba(0,0,0,.25),0 14px 34px -16px rgba(0,0,0,.6)}}
:root[data-theme="dark"]{--bg:#121530;--paper:#191d3b;--ink:#eceefa;--mut:#9aa0c3;--line:#2a2f55;--soft:#20254a;--glow:.32;--focus:#7aa7ff;
  --shadow:0 1px 2px rgba(0,0,0,.25),0 14px 34px -16px rgba(0,0,0,.6)}
*{box-sizing:border-box}
html{color-scheme:light dark}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);-webkit-font-smoothing:antialiased;overflow-x:hidden}
.aurora{position:absolute;inset:0 0 auto 0;height:620px;pointer-events:none;z-index:0;opacity:var(--glow);
  background:radial-gradient(38% 60% at 8% 0%,rgba(255,201,60,.55),transparent 70%),radial-gradient(34% 55% at 38% -6%,rgba(255,79,139,.42),transparent 72%),
  radial-gradient(40% 60% at 78% -4%,rgba(168,85,247,.42),transparent 72%),radial-gradient(30% 50% at 100% 10%,rgba(59,130,246,.38),transparent 70%);
  -webkit-mask-image:linear-gradient(#000 30%,transparent);mask-image:linear-gradient(#000 30%,transparent)}
.wrap{position:relative;z-index:1;max-width:1320px;margin:0 auto;padding:0 28px}
header{padding:52px 0 34px}
.topbar{display:flex;justify-content:flex-end;margin-bottom:6px}
.theme{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:8px;font:600 13px var(--body);color:var(--mut);
  background:color-mix(in srgb,var(--paper) 70%,transparent);border:1px solid var(--line);border-radius:999px;padding:6px 12px 6px 10px;backdrop-filter:blur(8px)}
.theme:hover{color:var(--ink)}
.theme svg{width:16px;height:16px}
.brand{display:flex;align-items:flex-end;gap:22px;flex-wrap:wrap}
.word{font:700 clamp(44px,6.4vw,72px)/.95 var(--display);letter-spacing:-.04em;margin:0}
.word .s{color:#ffb020}.word .l{color:#ff7a3d}.word .m{color:#ff4f8b}
.word small{display:block;font:600 clamp(18px,2.2vw,24px)/1.2 var(--display);letter-spacing:-.01em;color:var(--mut);margin-top:12px}
.bars{display:flex;align-items:flex-end;gap:4px;height:88px;margin-left:auto;padding-bottom:4px}
.bars i{display:block;width:9px;border-radius:3px 3px 1px 1px;transform-origin:bottom;animation:rise .9s cubic-bezier(.2,.8,.2,1) both}
.bars i:nth-child(n){animation-delay:calc(var(--d,0) * 1ms)}
@keyframes rise{from{transform:scaleY(.05)}to{transform:scaleY(1)}}
.lede{max-width:68ch;color:var(--mut);margin:20px 0 0}
.lede b{color:var(--ink);font-weight:600}
.stats{display:flex;flex-wrap:wrap;gap:10px;list-style:none;padding:0;margin:22px 0 0}
.stats li{display:flex;align-items:baseline;gap:7px;background:color-mix(in srgb,var(--paper) 78%,transparent);border:1px solid var(--line);
  border-radius:999px;padding:6px 14px;color:var(--mut);font-size:13.5px;backdrop-filter:blur(8px)}
.stats b{font:600 15px var(--display);color:var(--ink);letter-spacing:-.01em}
.rule{height:3px;border-radius:3px;background:var(--grad);opacity:.85;margin:0 0 26px}
.tabs{display:flex;gap:4px;flex-wrap:wrap;width:max-content;max-width:100%;margin:0 0 26px;padding:5px;background:var(--soft);border:1px solid var(--line);border-radius:16px}
.tabs button{font:600 15px var(--display);color:var(--mut);background:none;border:0;padding:9px 16px;cursor:pointer;display:flex;align-items:center;gap:9px;
  border-radius:12px;transition:background .15s,color .15s,box-shadow .15s}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected="true"]{color:var(--ink);background:var(--paper);box-shadow:var(--shadow),inset 0 -2px 0 var(--hue)}
.tabs .dot{width:10px;height:10px;border-radius:50%;background:var(--hue);box-shadow:0 0 0 3px color-mix(in srgb,var(--hue) 22%,transparent)}
.tabs .count{font:700 11.5px var(--body);background:color-mix(in srgb,var(--hue) 16%,transparent);color:var(--ink);border-radius:20px;padding:1px 8px}
.panel[hidden]{display:none}
.panel{animation:fade .25s ease both}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.intro{margin:0 0 18px}
.intro h2{font:600 28px/1.2 var(--display);letter-spacing:-.025em;margin:0 0 6px;display:flex;align-items:center;gap:12px}
.intro h2::before{content:"";width:12px;height:28px;border-radius:4px;background:linear-gradient(var(--hue),color-mix(in srgb,var(--hue) 45%,transparent))}
.intro p{margin:0;color:var(--mut);max-width:82ch}
.card{background:var(--paper);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}
.podium{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:16px;margin:0 0 16px}
.pod{position:relative;overflow:hidden;background:var(--paper);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);padding:18px 20px 20px}
.pod::before{content:"";position:absolute;inset:0 0 auto 0;height:90px;opacity:.5;pointer-events:none;
  background:radial-gradient(80% 100% at 0% 0%,color-mix(in srgb,var(--m) 45%,transparent),transparent 70%)}
.pod.gold{--m:#ffc93c}.pod.silver{--m:#b8bfd6}.pod.bronze{--m:#e0824a}
.podtop{position:relative;display:flex;align-items:center;gap:10px}
.podplace{font:600 12px var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.podname{position:relative;font:600 19px/1.25 var(--display);letter-spacing:-.015em;margin:14px 0 6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.podscore{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-top:14px}
.podscore b{font:700 38px/1 var(--display);letter-spacing:-.04em}
.podscore b::after{content:"%";font-size:.45em;color:var(--mut);margin-left:2px;letter-spacing:0}
.podscore span{color:var(--mut);font-weight:600;font-size:13px}
.pod .meter{width:100%;height:8px;margin-top:12px}
.scroll{overflow-x:auto}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums}
th{font:700 11.5px/1.25 var(--body);letter-spacing:.06em;text-transform:uppercase;color:var(--mut);text-align:right;padding:0;border-bottom:1px solid var(--line);vertical-align:bottom;background:var(--paper)}
th button{all:unset;display:block;cursor:pointer;white-space:nowrap;padding:16px 14px 13px;width:100%;box-sizing:border-box;text-align:inherit}
th button:hover{color:var(--ink)}
th[aria-sort] button{color:var(--ink)}
th[aria-sort] button::after{content:" \\2193";color:var(--hue)}th[aria-sort="ascending"] button::after{content:" \\2191"}
th.model,th.rank{text-align:left}
th:first-child{border-top-left-radius:20px}th:last-child{border-top-right-radius:20px}
td{padding:13px 14px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;vertical-align:middle}
tbody tr:last-child td{border-bottom:0}
td.rank{width:56px;text-align:left}
.rk{display:inline-grid;place-items:center;width:30px;height:30px;color:var(--mut);font-weight:600}
.medal{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;font:700 14px var(--display);color:#1c2240;
  box-shadow:inset 0 -2px 0 rgba(28,34,64,.18),0 2px 6px -2px rgba(28,34,64,.35)}
.medal.gold{background:linear-gradient(140deg,#ffe58f,#ffb020)}.medal.silver{background:linear-gradient(140deg,#f4f5fa,#aeb6cf)}
.medal.bronze{background:linear-gradient(140deg,#ffd0ad,#d9763c)}
td.model{text-align:left;white-space:normal;min-width:250px;position:sticky;left:0;background:var(--paper);z-index:1}
th.model{position:sticky;left:0;z-index:2}
.name{font:600 16px/1.3 var(--display);letter-spacing:-.01em}
.org{color:var(--mut);font-weight:500}
.tag{display:inline-block;margin-left:8px;font:600 11.5px var(--body);color:var(--mut);border:1px dashed var(--mut);border-radius:20px;padding:0 8px;vertical-align:2px}
.facts{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}
.facts span{font:600 12px var(--body);background:var(--soft);border:1px solid var(--line);border-radius:7px;padding:0 7px;color:var(--ink)}
.chip{display:inline-block;min-width:60px;text-align:center;font-weight:600;font-size:14.5px;padding:5px 9px;border-radius:10px;
  background:color-mix(in srgb,var(--hue) calc(var(--v) * var(--v) * 58%),transparent)}
td.num .sub{display:block;font:500 11.5px var(--body);color:var(--mut);margin-top:3px;text-align:center}
td.na,td.plain{color:var(--mut);font-weight:600}
td.na{text-align:center}
td.score{min-width:160px}
td.score .val{font:700 21px var(--display);letter-spacing:-.02em}
td.score .val::after{content:"%";font-size:.55em;color:var(--mut);margin-left:1px}
.meter{display:block;height:6px;border-radius:6px;margin-top:7px;margin-left:auto;width:128px;background:var(--soft);position:relative;overflow:hidden}
.meter::after{content:"";position:absolute;inset:0 auto 0 0;width:calc(var(--v) * 100%);border-radius:inherit;
  background:linear-gradient(90deg,color-mix(in srgb,var(--hue) 45%,var(--paper)),var(--hue))}
tr.divider td{background:var(--soft);color:var(--mut);font-size:13px;text-align:left;white-space:normal;padding:10px 16px;
  border-top:1px dashed var(--line)}
tr.divider span{display:inline-flex;gap:8px;align-items:center}
tr.divider span::before{content:"";width:8px;height:8px;border-radius:50%;border:1.5px dashed var(--mut)}
tr.unranked td.model .name{color:var(--mut)}
tr.unranked .chip{background:color-mix(in srgb,var(--mut) calc(4% + var(--v) * 22%),transparent)}
tr.unranked .meter::after{background:color-mix(in srgb,var(--mut) 55%,transparent)}
tbody tr:not(.divider):hover td{background-color:color-mix(in srgb,var(--hue) 5%,var(--paper))}
.settings{margin:36px 0 0}
.settings summary{cursor:pointer;font:600 17px var(--display);padding:6px 0;width:max-content;list-style:none;display:flex;align-items:center;gap:10px}
.settings summary::-webkit-details-marker{display:none}
.settings summary::before{content:"";width:8px;height:8px;border-right:2px solid var(--mut);border-bottom:2px solid var(--mut);transform:rotate(-45deg);transition:transform .2s}
.settings[open] summary::before{transform:rotate(45deg)}
.settings .scroll{margin-top:12px}
table.plain th,table.plain td{text-align:left;font-weight:500}
table.plain th{padding:14px}
footer{color:var(--mut);font-size:13.5px;padding:40px 0 56px;max-width:92ch}
footer::before{content:"";display:block;height:2px;width:120px;border-radius:2px;background:var(--grad);margin-bottom:22px}
footer p{margin:0 0 8px}
code{font:13px ui-monospace,Menlo,monospace;background:var(--soft);border-radius:6px;padding:1px 6px}
.empty{background:var(--paper);border:1px dashed var(--line);border-radius:20px;padding:56px 28px;text-align:center}
.empty h2{font:600 24px var(--display);margin:0 0 8px}.empty p{color:var(--mut);margin:0}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:6px}
@media (max-width:720px){.wrap{padding:0 16px}header{padding-top:24px}.bars{display:none}td,th button{padding-left:10px;padding-right:10px}
  td.model{min-width:180px}td.score{min-width:110px}.meter{width:80px}.tabs{width:100%}.tabs button{padding:8px 11px}.podscore b{font-size:32px}}
/*CHARTS*/
.tip{position:fixed;z-index:50;left:0;top:0;width:290px;pointer-events:none;opacity:0;transform:translateY(4px);transition:opacity .12s,transform .12s;
  background:color-mix(in srgb,var(--paper) 94%,transparent);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:14px;
  box-shadow:0 18px 40px -16px rgba(18,21,48,.45),0 2px 6px rgba(18,21,48,.08);padding:14px 16px 12px;font-size:13px;color:var(--ink)}
.tip.on{opacity:1;transform:none}
.tip::before{content:"";position:absolute;inset:0 0 auto 0;height:3px;border-radius:14px 14px 0 0;background:var(--hue)}
.tip .th{display:flex;align-items:center;gap:8px}
.tip .tn{font:600 15px/1.25 var(--display);letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tip .tn .org{font-weight:500}
.tip .medal{width:22px;height:22px;font-size:11.5px;flex:none}
.tip .tr{flex:none;font:700 11.5px var(--body);color:var(--mut);background:var(--soft);border-radius:20px;padding:1px 8px}
.tip .facts{margin-top:6px}
.tip .facts span{font-size:11px}
.tip .tk{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:12px 0 10px}
.tip .tk div{background:var(--soft);border-radius:10px;padding:6px 8px}
.tip .tk b{display:block;font:700 17px/1.1 var(--display);letter-spacing:-.02em}
.tip .tk small{color:var(--mut);font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.tip .ts{color:var(--mut);font-size:12px;margin:-2px 0 8px;display:flex;gap:6px;align-items:center}
.tip .ts::before{content:"";width:7px;height:7px;border-radius:50%;border:1.5px dashed var(--mut)}
.tip .tb{display:grid;grid-template-columns:minmax(0,1fr) 64px 38px;gap:4px 8px;align-items:center;font-size:12px}
.tip .tb span:nth-child(3n+1){color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tip .tb i{display:block;height:5px;border-radius:5px;background:var(--soft);position:relative;overflow:hidden}
.tip .tb i::after{content:"";position:absolute;inset:0 auto 0 0;width:calc(var(--v) * 100%);background:var(--hue);border-radius:inherit}
.tip .tb b{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.tip .tb em{font-style:normal;color:var(--mut);font-weight:500;font-size:10.5px;margin-left:2px}
@media (prefers-reduced-motion:reduce){.bars i,.panel{animation:none}}
</style></head><body>
<div class="aurora" aria-hidden="true"></div>
<div class="tip" id="tip" role="tooltip" aria-hidden="true"></div>
<div class="wrap">
<header><div class="topbar"><button type="button" class="theme" id="theme" aria-label="Colour theme"></button></div>
<div class="brand"><h1 class="word"><span class="s">s</span><span class="l">l</span><span class="m">m</span>bench<small>Leaderboard of small language models</small></h1>
<div class="bars" aria-hidden="true">%BARS%</div></div>
<p class="lede">Every number was measured locally with <b>slmbench</b> on public benchmarks run through Hugging Face's lighteval.</p>
<ul class="stats">%STATS%</ul></header>
<div class="rule" aria-hidden="true"></div>
<main>%MAIN%</main>
<footer><p>A mode's score is the mean of its benchmarks' headline metrics. Rows are comparable only when every benchmark of the mode was measured on the full set; those come first and get medals.
Rows marked “k of N measured” or “sample run” are listed below them and are indicative only.</p>
<p>Compare models measured on the same machine and settings (see “Run settings”). Click a column header to sort. Generated by <code>slmbench</code>.</p></footer>
</div>
<script>
(function(){
  var root=document.documentElement,btn=document.getElementById('theme');
  var icons={auto:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor"/></svg>',
    light:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    dark:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'};
  var names={auto:'Auto',light:'Light',dark:'Dark'},order=['auto','light','dark'],theme='auto';
  try{theme=localStorage.getItem('slmbench-theme')||'auto'}catch(e){}
  function apply(t){theme=order.indexOf(t)<0?'auto':t;if(theme==='auto')root.removeAttribute('data-theme');else root.setAttribute('data-theme',theme);
    btn.innerHTML=icons[theme]+names[theme];btn.title='Theme: '+names[theme]+' (click to change)'}
  apply(theme);
  btn.addEventListener('click',function(){apply(order[(order.indexOf(theme)+1)%3]);try{localStorage.setItem('slmbench-theme',theme)}catch(e){}});
  var tip=document.getElementById('tip'),cur=null;
  function esc(x){return String(x).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
  function pc(v){return v==null?'–':(v*100).toFixed(1)}
  function nm(n){var i=n.lastIndexOf('/');return i<0?esc(n):'<span class="org">'+esc(n.slice(0,i+1))+'</span>'+esc(n.slice(i+1))}
  function render(d){
    var medal=d.rank&&d.rank<=3?'<span class="medal '+['gold','silver','bronze'][d.rank-1]+'">'+d.rank+'</span>':'';
    var h='<div class="th">'+medal+'<span class="tn">'+nm(d.name)+'</span>'+(d.rank?'<span class="tr">#'+d.rank+' of '+d.of+'</span>':'')+'</div>';
    h+='<div class="facts">'+d.facts.map(function(f){return '<span>'+esc(f)+'</span>'}).join('')+'</div>';
    var k=[['Score',pc(d.score)]];if(d.p5!=null)k.push(['pass@5',pc(d.p5)]);
    k.push(['tok/s',d.tps==null?'–':d.tps]);if(d.pp)k.push(['prompt t/s',d.pp]);
    h+='<div class="tk">'+k.slice(0,3).map(function(x){return '<div><b>'+esc(x[1])+'</b><small>'+x[0]+'</small></div>'}).join('')+'</div>';
    if(d.status)h+='<div class="ts">'+esc(d.status)+'</div>';
    h+='<div class="tb">'+d.bench.map(function(b){return '<span title="'+esc(b[0])+'">'+esc(b[0])+'</span><i style="--v:'+b[1]+'"></i><b>'+pc(b[1])
      +(b[2]?'<em>n='+b[2]+'</em>':'')+'</b>'}).join('')+'</div>';
    return h}
  function place(x,y){var w=tip.offsetWidth,hh=tip.offsetHeight,vw=innerWidth,vh=innerHeight,
    l=x+18+w>vw-8?x-18-w:x+18,t=Math.min(Math.max(8,y-hh/2),vh-hh-8);tip.style.left=Math.max(8,l)+'px';tip.style.top=t+'px'}
  function open(g,x,y){if(cur!==g){cur=g;try{tip.innerHTML=render(JSON.parse(g.dataset.tip))}catch(e){return}
      tip.style.setProperty('--hue',getComputedStyle(g.closest('.panel')).getPropertyValue('--hue'))}
    tip.classList.add('on');tip.setAttribute('aria-hidden','false');place(x,y)}
  function close(){cur=null;tip.classList.remove('on');tip.setAttribute('aria-hidden','true')}
  document.addEventListener('pointermove',function(e){var g=e.target.closest&&e.target.closest('.hit[data-tip]');if(g)open(g,e.clientX,e.clientY);else if(cur&&e.pointerType==='mouse')close()});
  document.addEventListener('pointerdown',function(e){var g=e.target.closest&&e.target.closest('.hit[data-tip]');if(!g)close();else open(g,e.clientX,e.clientY)});
  document.addEventListener('focusin',function(e){var g=e.target.closest&&e.target.closest('.hit[data-tip]');if(g){var r=g.getBoundingClientRect();open(g,r.right,r.top+r.height/2)}});
  document.addEventListener('focusout',close);
  document.addEventListener('keydown',function(e){if(e.key==='Escape')close()});
  addEventListener('scroll',function(){if(cur&&document.activeElement!==cur)close()},{passive:true});
  document.querySelectorAll('.bars i').forEach(function(b,i){b.style.setProperty('--d',i*35)});
  var tabs=[].slice.call(document.querySelectorAll('[role=tab]'));
  function show(mode,focus){tabs.forEach(function(t){var on=t.dataset.mode===mode;t.setAttribute('aria-selected',on);t.tabIndex=on?0:-1;
    document.getElementById('p-'+t.dataset.mode).hidden=!on;if(on&&focus)t.focus()});
    try{history.replaceState(null,'','#'+mode)}catch(e){}}
  tabs.forEach(function(t,i){t.addEventListener('click',function(){show(t.dataset.mode)});
    t.addEventListener('keydown',function(e){var d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;if(d){e.preventDefault();show(tabs[(i+d+tabs.length)%tabs.length].dataset.mode,true)}})});
  if(tabs.length){var h=location.hash.slice(1);show(tabs.some(function(t){return t.dataset.mode===h})?h:tabs[0].dataset.mode)}
  document.querySelectorAll('.panel table').forEach(function(t){
    var ths=[].slice.call(t.tHead.rows[0].cells);
    ths.forEach(function(th,i){th.querySelector('button').addEventListener('click',function(){
      var asc=th.getAttribute('aria-sort')==='descending';ths.forEach(function(x){x.removeAttribute('aria-sort')});
      th.setAttribute('aria-sort',asc?'ascending':'descending');
      var body=t.tBodies[0],rows=[].slice.call(body.rows).filter(function(r){return !r.classList.contains('divider')});
      var div=body.querySelector('tr.divider');if(div)div.remove();
      rows.sort(function(a,b){var x=a.cells[i].dataset.v,y=b.cells[i].dataset.v,nx=parseFloat(x),ny=parseFloat(y);
        var r=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));if(i<2)r=-r;return asc?r:-r});
      rows.forEach(function(r){body.appendChild(r)})})})});
})();
</script></body></html>"""
