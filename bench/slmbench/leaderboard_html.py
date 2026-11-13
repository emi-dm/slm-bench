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
MODE_HUE = {"general": "#ff8a3d", "code": "#3b82f6", "tools": "#ec4899", "long": "#8b5cf6"}
GRADIENT = ["#ffc93c", "#ffa83a", "#ff8a3d", "#ff6a5c", "#ff4f8b", "#d946ef", "#a855f7", "#8b5cf6", "#6366f1", "#3b82f6"]
BARS = [3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 22]


def _pct(v, d=1) -> str:
    return "–" if v is None else f"{v * 100:.{d}f}"


def _cell(v, sub: str = "", strong: bool = False) -> str:
    if v is None:
        return '<td class="num na" data-v="-1"><span aria-label="not measured">–</span></td>'
    s = f'<span class="sub">{sub}</span>' if sub else ""
    cls = "num score" if strong else "num"
    bar = f'<span class="meter" style="--v:{v:.4f}"></span>' if strong else ""
    return f'<td class="{cls}" data-v="{v:.4f}" style="--v:{v:.4f}"><span class="val">{_pct(v)}</span>{bar}{s}</td>'


def _speed(sp: dict, key: str) -> str:
    v = sp.get(key)
    return f'<td class="num plain" data-v="{v or 0}">{v if v else "–"}</td>'


def _charts(rows, mode: str, title: str) -> str:
    from . import leaderboard as lb
    from .leaderboard_charts import bars, scatter
    b = bars(rows, f"{title} score by model")
    s = scatter(rows, lambda e: lb.mode_speed(e, mode), f"{title} score against speed")
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
        heads = [("Rank", "rank"), ("Model", "model"), ("Score", "num")] + ([("Score at 5", "num")] if p5 else []) \
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
                body.append(f'<tr class="divider"><td colspan="{len(heads)}">{note}</td></tr>')
            medal = ["gold", "silver", "bronze"][i] if sc["ranked"] and i < 3 else ""
            rank = f'<span class="medal {medal}">{i + 1}</span>' if medal else f'<span class="rk">{i + 1}</span>'
            tag = "" if sc["ranked"] else (f'<span class="tag">{sc["n"]} of {sc["of"]} measured</span>' if not sc["complete"]
                                           else '<span class="tag">sample run</span>')
            facts = [x for x in (lb.params_txt(m) if m.get("params_b") else "", m.get("quant", ""),
                                 f"{m['size_gb']} GB" if m.get("size_gb") else "") if x]
            model = (f'<td class="model" data-v="{escape(e["name"].lower())}"><span class="name">{escape(e["name"])}</span>{tag}'
                     f'<span class="facts">{"".join(f"<span>{escape(str(x))}</span>" for x in facts)}</span>'
                     + (f'<span class="hw">{escape(m["hardware"])}</span>' if m.get("hardware") else "") + "</td>")
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
        tabs.append(f'<button type="button" role="tab" id="tab-{mode}" aria-controls="p-{mode}" data-mode="{mode}" '
                    f'style="--hue:{MODE_HUE.get(mode, "#ff8a3d")}"><span class="dot"></span>{escape(title)}'
                    f'<span class="count">{len(rows)}</span></button>')
        lead = (f"{n_ranked} of {len(rows)} rows have every benchmark measured on the full set." if n_ranked != len(rows)
                else f"All {len(rows)} rows have every benchmark measured on the full set.")
        panels.append(
            f'<section class="panel" role="tabpanel" id="p-{mode}" aria-labelledby="tab-{mode}" style="--hue:{MODE_HUE.get(mode, "#ff8a3d")}">'
            f'<div class="intro"><h2>{escape(title)}</h2><p>{escape(info.get("desc_en", ""))} {lead} {how}</p></div>'
            + _charts(rows, mode, title) +
            f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>')

    details = []
    for e in sorted(entries, key=lambda x: x["name"].lower()):
        m, b = e["meta"], e["benchmarks"].values()
        temps = sorted({x.get("temperature") for x in b if x.get("temperature") is not None})
        budgets = sorted({x.get("reasoning_budget", -1) for x in b})
        maxt = sorted({x.get("max_tokens") for x in b if x.get("max_tokens")})
        pk = sorted({f"{x['samples']} at T={x.get('sample_temperature')}" if (x.get("samples") or 1) > 1 else "1"
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
    if entries:
        main = (f'<nav class="tabs" role="tablist" aria-label="Evaluation mode">{"".join(tabs)}</nav>{"".join(panels)}'
                f'<details class="settings"><summary>Run settings for each row</summary><div class="scroll"><table class="plain">'
                f'<thead><tr>{"".join(f"<th scope=col>{h}</th>" for h in det_head)}</tr></thead><tbody>{"".join(details)}</tbody>'
                f'</table></div></details>')
    else:
        main = ('<div class="empty"><h2>No models yet</h2><p>Run <code>slmbench run</code> and every benchmark you measure will '
                'show up here, one row per model and quantization.</p></div>')
    stats = (f"{n_models} model{'s' if n_models != 1 else ''}, {len(entries)} row{'s' if len(entries) != 1 else ''} "
             f"(model and quantization). Updated {today}.") if entries else f"Updated {today}."
    from .leaderboard_charts import CSS as CHART_CSS
    return TEMPLATE.replace("/*CHARTS*/", CHART_CSS).replace("%BARS%", bars).replace("%STATS%", stats).replace("%MAIN%", main)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLM Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f6f7fb;--paper:#ffffff;--ink:#1c2240;--mut:#636a8c;--line:#e4e7f1;--soft:#eef0f7;--tint:20%;--focus:#3b82f6;
--display:"Lexend",ui-rounded,"SF Pro Rounded",system-ui,sans-serif;--body:"Manrope",system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121530;--paper:#191d3b;--ink:#eceefa;--mut:#9aa0c3;--line:#2a2f55;--soft:#20254a;--tint:34%;--focus:#7aa7ff}}
:root[data-theme="dark"]{--bg:#121530;--paper:#191d3b;--ink:#eceefa;--mut:#9aa0c3;--line:#2a2f55;--soft:#20254a;--tint:34%;--focus:#7aa7ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);-webkit-font-smoothing:antialiased}
.wrap{max-width:1320px;margin:0 auto;padding:0 28px}
header{padding:56px 0 30px}
.brand{display:flex;align-items:flex-end;gap:22px;flex-wrap:wrap}
.word{font:700 clamp(40px,6vw,64px)/.95 var(--display);letter-spacing:-.035em;margin:0}
.word .s{color:#ffb020}.word .l{color:#ff7a3d}.word .m{color:#ff4f8b}
.word small{display:block;font:600 clamp(18px,2.2vw,24px)/1.2 var(--display);letter-spacing:-.01em;color:var(--mut);margin-top:10px}
.bars{display:flex;align-items:flex-end;gap:4px;height:88px;margin-left:auto;padding-bottom:4px}
.bars i{display:block;width:9px;border-radius:3px 3px 1px 1px;transform-origin:bottom;animation:rise .9s cubic-bezier(.2,.8,.2,1) both}
.bars i:nth-child(n){animation-delay:calc(var(--d,0) * 1ms)}
@keyframes rise{from{transform:scaleY(.05)}to{transform:scaleY(1)}}
.lede{max-width:68ch;color:var(--mut);margin:18px 0 0}
.lede b{color:var(--ink);font-weight:600}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 18px;border-bottom:1px solid var(--line)}
.tabs button{font:600 15px var(--display);color:var(--mut);background:none;border:0;padding:12px 14px 13px;cursor:pointer;display:flex;align-items:center;gap:9px;
  border-bottom:3px solid transparent;margin-bottom:-1px;border-radius:8px 8px 0 0}
.tabs button:hover{color:var(--ink);background:var(--soft)}
.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--hue)}
.tabs .dot{width:10px;height:10px;border-radius:50%;background:var(--hue)}
.tabs .count{font:600 12px var(--body);background:var(--soft);color:var(--mut);border-radius:20px;padding:1px 8px}
.panel[hidden]{display:none}
.intro{display:grid;grid-template-columns:minmax(0,1fr);margin:0 0 16px}
.intro h2{font:600 26px/1.2 var(--display);letter-spacing:-.02em;margin:6px 0 6px}
.intro p{margin:0;color:var(--mut);max-width:80ch}
.scroll{overflow-x:auto;background:var(--paper);border:1px solid var(--line);border-radius:18px}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums}
th{font:600 12.5px/1.25 var(--body);color:var(--mut);text-align:right;padding:0;border-bottom:1px solid var(--line);vertical-align:bottom;background:var(--paper)}
th button{all:unset;display:block;cursor:pointer;white-space:nowrap;padding:14px 14px 12px;width:100%;box-sizing:border-box;text-align:inherit}
th button:hover{color:var(--ink)}
th[aria-sort] button::after{content:" \\2193";color:var(--hue)}th[aria-sort="ascending"] button::after{content:" \\2191"}
th.model,th.rank{text-align:left}
td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;vertical-align:middle}
tbody tr:last-child td{border-bottom:0}
td.rank{width:56px;text-align:left}
.rk{display:inline-grid;place-items:center;width:30px;height:30px;color:var(--mut);font-weight:600}
.medal{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;font:700 14px var(--display);color:#1c2240}
.medal.gold{background:linear-gradient(140deg,#ffe07a,#ffb020)}.medal.silver{background:linear-gradient(140deg,#eef0f7,#b8bfd6)}
.medal.bronze{background:linear-gradient(140deg,#ffc9a0,#e0824a)}
td.model{text-align:left;white-space:normal;min-width:240px;position:sticky;left:0;background:var(--paper);z-index:1}
th.model{position:sticky;left:0;z-index:2}
.name{font:600 16px/1.3 var(--display);letter-spacing:-.01em}
.tag{display:inline-block;margin-left:8px;font:600 11.5px var(--body);color:var(--mut);border:1px dashed var(--mut);border-radius:20px;padding:0 8px;vertical-align:2px}
.facts{display:flex;gap:6px;margin-top:5px;flex-wrap:wrap}
.facts span{font:600 12px var(--body);background:var(--soft);border-radius:6px;padding:1px 7px;color:var(--ink)}
.hw{display:block;color:var(--mut);font-size:12.5px;margin-top:4px}
td.num{background:linear-gradient(to top,color-mix(in srgb,var(--hue) var(--tint),transparent) calc(var(--v,0) * 100%),transparent 0);box-shadow:inset 3px 0 var(--paper),inset -3px 0 var(--paper);font-weight:600}
td.num .val{font-size:15px}
td.num .sub{display:block;font:500 11.5px var(--body);color:var(--mut);margin-top:1px}
td.na,td.plain{background:none;color:var(--mut);font-weight:500}
td.score{background:none;min-width:150px}
td.score .val{font:700 20px var(--display);letter-spacing:-.02em}
.meter{display:block;height:6px;border-radius:6px;margin-top:6px;margin-left:auto;width:120px;
  background:linear-gradient(90deg,var(--hue) calc(var(--v) * 100%),var(--soft) 0)}
tr.divider td{background:var(--soft);color:var(--mut);font-size:13px;text-align:left;white-space:normal;padding:9px 14px}
tr.unranked td.model .name{color:var(--mut)}
tbody tr.ranked:hover td,tbody tr.unranked:hover td{box-shadow:inset 0 0 0 9999px color-mix(in srgb,var(--ink) 4%,transparent)}
.settings{margin:34px 0 0}
.settings summary{cursor:pointer;font:600 16px var(--display);padding:6px 0;width:max-content}
.settings .scroll{margin-top:12px}
table.plain th,table.plain td{text-align:left;font-weight:500}
table.plain th{padding:12px 14px}
footer{color:var(--mut);font-size:13.5px;padding:34px 0 56px;max-width:90ch}
footer p{margin:0 0 8px}
code{font:13px ui-monospace,Menlo,monospace;background:var(--soft);border-radius:6px;padding:1px 6px}
.empty{background:var(--paper);border:1px dashed var(--line);border-radius:18px;padding:56px 28px;text-align:center}
.empty h2{font:600 24px var(--display);margin:0 0 8px}.empty p{color:var(--mut);margin:0}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:6px}
@media (max-width:720px){.wrap{padding:0 16px}header{padding-top:32px}.bars{display:none}td,th button{padding-left:10px;padding-right:10px}
  td.model{min-width:180px}td.score{min-width:110px}.meter{width:80px}}
/*CHARTS*/
@media (prefers-reduced-motion:reduce){.bars i{animation:none}}
</style></head><body>
<div class="wrap">
<header><div class="brand"><h1 class="word"><span class="s">s</span><span class="l">l</span><span class="m">m</span>bench<small>Leaderboard of small language models</small></h1>
<div class="bars" aria-hidden="true">%BARS%</div></div>
<p class="lede">Every number was measured locally with <b>slmbench</b> on public benchmarks run through Hugging Face's lighteval. %STATS%</p></header>
<main>%MAIN%</main>
<footer><p>A mode's score is the mean of its benchmarks' headline metrics. Rows are comparable only when every benchmark of the mode was measured on the full set; those come first and get medals.
Rows marked “k of N measured” or “sample run” are listed below them and are indicative only.</p>
<p>Compare models measured on the same machine and settings (see “Run settings”). Click a column header to sort. Generated by <code>slmbench</code>.</p></footer>
</div>
<script>
(function(){
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
