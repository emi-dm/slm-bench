"""Leaderboard: guarda el resultado de cada modelo evaluado y genera `leaderboard.md`, `leaderboard/leaderboard.html` y
`leaderboard/leaderboard.svg`.

Datos: un JSON por fila en `<dir>/leaderboard/entries/<id>.json`. Cada fila es **id del modelo + cuantización**: el id es el que devuelve
el servidor (sin el sufijo de quant ni «-GGUF») y la quant la da llama-server (o se deduce del id). Así, un mismo modelo medido en Q4_K_M y
Q8_0 ocupa dos filas, y el leaderboard añade una tabla que compara sus cuantizaciones.

Cada fila guarda la métrica principal de cada benchmark (catálogo). La puntuación de un modo (General, Programación, Herramientas,
Contexto largo) es la media de sus benchmarks medidos; solo es comparable (y lleva medalla) si se midieron todos sobre el conjunto
completo. Las ejecuciones parciales (`--max-samples`) se marcan con n= y nunca sustituyen a un resultado completo ya guardado.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import re
import subprocess
import urllib.request
from html import escape
from pathlib import Path

from .config import Settings



# ---------------------------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------------------------
def default_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    if (root / "bench" / "slmbench").is_dir():
        return root
    from .engine import data_dir
    return data_dir() / "leaderboard"


def paths(s: Settings | None = None) -> dict:
    base = Path(s.leaderboard_dir).expanduser() if s is not None and getattr(s, "leaderboard_dir", "") else default_dir()
    sub = base / "leaderboard"
    return {"base": base, "md": base / "leaderboard.md", "dir": sub, "entries": sub / "entries", "html": sub / "leaderboard.html", "svg": sub / "leaderboard.svg"}


QUANT_RE = re.compile(r"(?i)(?:^|(?<=[-_.:@ /]))((?:UD-)?I?Q\d(?:_[A-Z0-9]+)*|BF16|FP16|F16|F32|MXFP4|\d+-?bit)(?=$|[-_.:@ ])")


def split_quant(model_id: str, quant: str | None = None) -> tuple[str, str]:
    """('LiquidAI/LFM3-1.7B-GGUF:Q4_K_M', None) -> ('LiquidAI/LFM3-1.7B', 'Q4_K_M'). La quant del servidor manda si existe."""
    base = model_id or ""
    m = QUANT_RE.search(base)
    if m:
        quant = quant or m.group(1).upper()
        base = (base[: m.start(1)] + base[m.end(1):])
    base = re.sub(r"(?i)[-_.]gguf\b", "", base)
    base = re.sub(r"[-_.:@ ]+$", "", re.sub(r"[-_.:@ ]{2,}", "-", base)).strip("-_.:@ ") or model_id
    return base, (quant or "")


def entry_identity(model_id: str, quant: str | None) -> tuple[str, str, str]:
    """(id del fichero, nombre del modelo, quant). Cada fila del leaderboard es modelo + quant."""
    from .servers import clean_model_id
    mid = clean_model_id(model_id)
    if ":" in mid and "/" in mid.split(":")[0]:            # "org/repo-GGUF:file-stem" from a cache path: org/repo + quant from the file
        repo, stem = mid.split(":", 1)
        mid = repo + ":" + (split_quant(stem, quant)[1] or stem)
    base, q = split_quant(mid, quant)
    return slugify(f"{base}-{q}" if q else base), base, q


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "model"


# ---------------------------------------------------------------------------------------------
# Metadatos del modelo y de la máquina
# ---------------------------------------------------------------------------------------------
def _get_json(url: str, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return {}


def hardware() -> str:
    try:
        if platform.system() == "Darwin":
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
            return f"{chip} · {mem / 1024 ** 3:.0f} GB"
    except Exception:
        pass
    return platform.processor() or platform.machine()


def model_meta(base_url: str, model_id: str) -> dict:
    """Metadatos del modelo (los que exponga el servidor) y de la máquina."""
    from .servers import model_meta as server_meta
    out = server_meta(base_url, model_id, timeout=4)
    out["hardware"] = hardware()
    return out


# ---------------------------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------------------------
def migrate(s: Settings | None = None):
    """Las entradas antiguas se guardaban por etiqueta: las renombra al id del modelo (fusionando si hace falta)."""
    d = paths(s)["entries"]
    if not d.is_dir():
        return
    for f in sorted(d.glob("*.json")):
        try:
            e = json.loads(f.read_text())
        except Exception:
            continue
        mid = e.get("model_id") or e.get("name")
        if not mid:
            continue
        eid, base, q = entry_identity(mid, (e.get("meta") or {}).get("quant"))
        if e.get("id") == eid and e.get("name") == base and (e.get("meta") or {}).get("quant", "") == q:
            continue
        if e.get("name") and e["name"] not in (mid, base):
            e.setdefault("meta", {})["label"] = e["name"]
        e["id"], e["name"] = eid, base
        if q:
            e.setdefault("meta", {})["quant"] = q
        target = d / f"{e['id']}.json"
        if target.exists() and target != f:
            old = json.loads(target.read_text())
            newer, older = (e, old) if (e.get("updated") or "") >= (old.get("updated") or "") else (old, e)
            for k in ("suites", "lighteval", "benchmarks"):
                newer[k] = {**older.get(k, {}), **newer.get(k, {})}
            newer["meta"] = {**older.get("meta", {}), **newer.get("meta", {})}
            e = newer
        target.write_text(json.dumps(e, indent=1, ensure_ascii=False))
        if target != f:
            f.unlink()


def load_entries(s: Settings | None = None) -> list[dict]:
    migrate(s)
    d = paths(s)["entries"]
    out = []
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                out.append(_upgrade(json.loads(f.read_text())))
            except Exception:
                continue
    return out


def save_entry(e: dict, s: Settings | None = None):
    d = paths(s)["entries"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{e['id']}.json").write_text(json.dumps(e, indent=1, ensure_ascii=False))


def remove_entry(entry_id: str, s: Settings | None = None) -> bool:
    f = paths(s)["entries"] / f"{entry_id}.json"
    if f.exists():
        f.unlink()
        rebuild(s)
        return True
    return False


def _upgrade(e: dict) -> dict:
    """Formato antiguo (suites propias + «lighteval») -> formato actual (solo «benchmarks»)."""
    from .benchmarks import catalog as cat
    if "benchmarks" not in e:
        e["benchmarks"] = {}
        c = cat.catalog()
        for task, r in (e.get("lighteval") or {}).items():
            key = cat.spec_to_key(task)
            if key:
                n = r.get("n") or 0
                e["benchmarks"][key] = {"metric": r.get("metric"), "value": r.get("value"), "stderr": r.get("stderr"),
                                        "n": n or c[key]["n"], "full": not n, "date": r.get("date")}
    e.pop("suites", None)
    e.pop("lighteval", None)
    e.setdefault("speed", {})
    return e


def _new_entry(entry_id: str, name: str, model_id: str) -> dict:
    return {"id": entry_id, "name": name, "model_id": model_id, "created": dt.date.today().isoformat(), "updated": None,
            "meta": {}, "speed": {}, "benchmarks": {}}


def record(s: Settings, meta: dict, summary: dict) -> dict | None:
    """Registra una ejecución (resultados de lighteval) en la fila modelo + quant."""
    from .benchmarks import catalog as cat
    from .engine import headline
    if meta.get("cancelled") or not summary or not getattr(s, "leaderboard_auto", True):
        return None
    # the server's model data as read when the run started (quant, size…); older runs: ask the server now
    info = {**meta["server"], "hardware": hardware()} if meta.get("server") else model_meta(s.base_url, meta["model"])
    if info.get("lora") and not info.get("lora_base"):    # a server started elsewhere: the adapter's base from its card
        try:
            from .lora import base_model, repo_of
            repo = info.get("lora_repo") or repo_of(info.get("lora_path", ""))
            info["lora_base"] = base_model(repo) if repo else ""
        except Exception:  # noqa: BLE001 - offline: the ranking just can't compare it with its base
            pass
    row_model = meta["model"]
    if info.get("lora") and info.get("lora_base") and _norm(entry_identity(meta["model"], None)[1]) == _norm(info["lora_base"]):
        # the server names the model after its base (Unsloth Studio does): the row is the adapter's, not the base's
        from .lora import repo_of
        repo = info.get("lora_repo") or repo_of(info.get("lora_path", ""))
        row_model = re.sub(r"(?i)[-_.]gguf$", "", repo) if repo else Path(str(info.get("lora_path") or info["lora"])).stem
    eid, name, quant = entry_identity(row_model, info.get("quant"))
    e = next((x for x in load_entries(s) if x["id"] == eid), None) or _new_entry(eid, name, row_model)
    e["model_id"] = row_model                         # the row's identity (the adapter's when the server names the base)
    if row_model != meta["model"]:
        e["meta"]["served_id"] = meta["model"]
    e["meta"].update(info)
    if quant:
        e["meta"]["quant"] = quant
    c = cat.catalog()
    meta["leaderboard_kept"] = []
    for task, metrics in summary.items():
        key = cat.spec_to_key(task) or task.split("|")[0]
        mname, m = headline(task, metrics)
        if not m:
            continue
        n_full = c.get(key, {}).get("n")
        ms = meta.get("max_samples") or 0
        n = min(ms, n_full) if ms and n_full else (ms or n_full)
        full = not ms or (n_full is not None and ms >= n_full)
        old = e["benchmarks"].get(key)
        if old and not full and (old.get("full") or (old.get("n") or 0) > (n or 0)):
            meta["leaderboard_kept"].append(key)        # a quick partial run never replaces a full (or bigger) result
            continue
        e["benchmarks"][key] = {"metric": mname, "value": m["value"], "stderr": m.get("stderr"), "n": n, "full": full,
                                "date": meta["date"][:10], "temperature": meta.get("temperature"), "reasoning_budget": meta.get("reasoning_budget", -1),
                                "max_tokens": cat.effective_max_tokens(key, meta.get("max_new_tokens") or 0) or meta.get("max_new_tokens")}
        if c.get(key, {}).get("mode") == "agent":           # agent mode: which harness, its version and how the patches went
            e["benchmarks"][key].update({
                "harness": c[key].get("harness"), "harness_version": (meta.get("harness_versions") or {}).get(c[key].get("harness")),
                "patch_applied": (metrics.get("patch_applied") or {}).get("value"),
                "submitted": (metrics.get("submitted") or {}).get("value"),
                "step_limit": meta.get("step_limit"), "dataset": (meta.get("dataset") or {}).get("revision"),
                "arch": meta.get("arch")})
            st = (meta.get("agent_stats") or {}).get(task) or {}
            if st:                                           # cost per task: model calls and tokens
                e["benchmarks"][key].update({"turns": st.get("turns_mean"), "tokens": st.get("tokens_mean")})
        if c.get(key, {}).get("passk"):                     # programación: pass@5 y cómo se muestreó
            p5 = metrics.get("pass@5") or {}
            e["benchmarks"][key].update({"pass5": p5.get("value"), "pass5_stderr": p5.get("stderr"),
                                         "samples": meta.get("passk_n", 1)})
    # speed: per mode (a long-context prompt is much slower to generate after); the headline tok/s excludes long context.
    # With fewer parallel requests each one is faster, so the lowest worker count measured wins.
    def better(old):
        return not old or old.get("workers") is None or s.workers <= old["workers"]
    by_mode = meta.get("speed_by_mode") or {}
    for md, sp in by_mode.items():
        if sp.get("gen_tps") and better(e.setdefault("speed_modes", {}).get(md)):
            e["speed_modes"][md] = {**sp, "workers": s.workers, "date": meta["date"][:10]}
    if by_mode:
        gen = next((by_mode[md] for md in ("general", "code", "tools") if (by_mode.get(md) or {}).get("gen_tps")), None)
    else:
        gen = {"gen_tps": meta["gen_tps"], "prompt_tps": meta.get("prompt_tps")} if meta.get("gen_tps") else None
    if gen and better(e.get("speed")):
        e["speed"] = {"gen_tps": gen["gen_tps"], "prompt_tps": gen.get("prompt_tps"), "workers": s.workers, "date": meta["date"][:10]}
    e["updated"] = meta["date"][:10]
    save_entry(e, s)
    rebuild(s)
    return e


# ---------------------------------------------------------------------------------------------
# Puntuaciones
# ---------------------------------------------------------------------------------------------
def modes() -> list[tuple[str, str]]:
    """(mode, English title) of every evaluation mode in the catalog (general, code, tools, long…)."""
    from .benchmarks import catalog as cat
    return [(m, info.get("en", m)) for m, info in cat.MODES.items()]


MODE_CLASS = {"general": "gen", "code": "code", "tools": "tools", "long": "long", "agent": "agent"}


def mode_keys(mode: str) -> list[str]:
    from .benchmarks import catalog as cat
    return cat.mode_benchmarks(mode)


def mode_score(e: dict, mode: str) -> dict:
    keys = mode_keys(mode)
    vals = [e["benchmarks"][k]["value"] for k in keys if k in e["benchmarks"] and e["benchmarks"][k].get("value") is not None]
    present = [k for k in keys if k in e["benchmarks"]]
    partial = [k for k in present if not e["benchmarks"][k].get("full")]
    complete = len(present) == len(keys)
    return {"score": sum(vals) / len(vals) if vals else None, "n": len(present), "of": len(keys),
            "complete": complete, "partial": bool(partial), "ranked": complete and not partial}


def mode_pass5(e: dict, mode: str) -> float | None:
    """Mean pass@5 of the mode, only when every measured benchmark has it (otherwise it would mix different sets)."""
    rs = [e["benchmarks"][k] for k in mode_keys(mode) if k in e["benchmarks"]]
    vals = [r.get("pass5") for r in rs]
    return sum(vals) / len(vals) if vals and all(v is not None for v in vals) else None


def mode_speed(e: dict, mode: str) -> dict:
    """tok/s measured on this mode's benchmarks; falls back to the headline speed (except for long context)."""
    sp = (e.get("speed_modes") or {}).get(mode)
    return sp or ({} if mode == "long" else e.get("speed") or {})


def ranked(entries: list[dict], mode: str) -> list[tuple[dict, dict]]:
    """Rows with something measured in the mode. Order: every benchmark measured on the full set first (these are the only
    comparable scores and get medals), then complete but partial (n=) runs, then rows missing benchmarks."""
    rows = [(e, mode_score(e, mode)) for e in entries]
    rows = [r for r in rows if r[1]["n"]]
    rows.sort(key=lambda r: (0 if r[1]["ranked"] else 1 if r[1]["complete"] else 2, -(r[1]["score"] or 0), r[0]["name"].lower()))
    return rows


# ---------------------------------------------------------------------------------------------
# Harnesses (agent mode): the same benchmark driven by different agents
# ---------------------------------------------------------------------------------------------
def harness_keys() -> list[str]:
    from .benchmarks import catalog as cat
    return cat.mode_benchmarks("agent")


def has_harnesses(entries: list[dict]) -> bool:
    return any(k in e["benchmarks"] for e in entries for k in harness_keys())


def harness_table(entries: list[dict]) -> list[dict]:
    """One row per harness: models measured, mean resolved over them, the best model, its version and how it acts."""
    from .benchmarks import catalog as cat
    c = cat.catalog()
    out = []
    for k in harness_keys():
        got = [(e, e["benchmarks"][k]) for e in entries if (e["benchmarks"].get(k) or {}).get("value") is not None]
        vals = [r["value"] for _, r in got]
        best = max(got, key=lambda x: (x[1]["value"], x[1].get("full", False)), default=None)
        turns = [r["turns"] for _, r in got if r.get("turns") is not None]
        toks = [r["tokens"] for _, r in got if r.get("tokens") is not None]
        out.append({"key": k, "name": c[k].get("harness_name", k), "harness": c[k].get("harness"),
                    "turns": sum(turns) / len(turns) if turns else None, "tokens": sum(toks) / len(toks) if toks else None,
                    "note_en": c[k].get("harness_note_en", ""), "note_es": c[k].get("harness_note_es", ""),
                    "version": next((r.get("harness_version") for _, r in sorted(got, key=lambda x: x[1].get("date") or "", reverse=True)
                                     if r.get("harness_version")), ""),
                    "models": len(got), "mean": sum(vals) / len(vals) if vals else None,
                    "best": ({"name": best[0]["name"], "quant": best[0]["meta"].get("quant", ""), "value": best[1]["value"],
                              "full": best[1].get("full"), "n": best[1].get("n")} if best else None)})
    return out


def _harness_results(e: dict) -> list[tuple[str, float]]:
    return [(k, e["benchmarks"][k]["value"]) for k in harness_keys() if (e["benchmarks"].get(k) or {}).get("value") is not None]


def best_harness(e: dict) -> tuple[str, float] | None:
    """The harness with which this model resolves the most tasks: (benchmark key, value); None if none resolves any."""
    got = [x for x in _harness_results(e) if x[1] > 0]
    return max(got, key=lambda x: x[1]) if got else None


def harness_rows(entries: list[dict]) -> list[dict]:
    """Models with at least one harness measured, best first (by their best harness)."""
    rows = [e for e in entries if _harness_results(e)]
    return sorted(rows, key=lambda e: (-max(v for _k, v in _harness_results(e)), e["name"].lower()))


# ---------------------------------------------------------------------------------------------
# LoRA adapters: each one against its base model
# ---------------------------------------------------------------------------------------------
def has_adapters(entries: list[dict]) -> bool:
    return any(e["meta"].get("lora") for e in entries)


def _norm(name: str) -> str:
    return re.sub(r"(?i)[-_.]gguf$", "", (name or "").strip()).lower()


def find_base(entries: list[dict], e: dict) -> dict | None:
    """The leaderboard row of an adapter's base model: same name (without -GGUF), no adapter, same quant if there is one."""
    want = _norm(e["meta"].get("lora_base", ""))
    if not want:
        return None
    rows = [x for x in entries if x is not e and not x["meta"].get("lora") and _norm(x["name"]) == want]
    same = [x for x in rows if x["meta"].get("quant") == e["meta"].get("quant")]
    return (same or rows or [None])[0]


def adapter_rows(entries: list[dict]) -> list[dict]:
    """One row per model served with a LoRA adapter: its score per mode and the gain over its base model, both measured
    on the benchmarks that the two rows share (so the difference compares like with like). Ranked by the mean gain;
    adapters whose base isn't in the leaderboard go after, by their score."""
    out = []
    for e in entries:
        if not e["meta"].get("lora"):
            continue
        base = find_base(entries, e)
        per = {}
        for mode, title in modes():
            keys = [k for k in mode_keys(mode) if (e["benchmarks"].get(k) or {}).get("value") is not None]
            if not keys:
                continue
            common = [k for k in keys if base and (base["benchmarks"].get(k) or {}).get("value") is not None]
            use = common or keys
            score = sum(e["benchmarks"][k]["value"] for k in use) / len(use)
            delta = (sum(e["benchmarks"][k]["value"] - base["benchmarks"][k]["value"] for k in common) / len(common)) if common else None
            per[mode] = {"title": title, "score": score, "delta": delta, "n": len(use),
                         "partial": any(not e["benchmarks"][k].get("full") for k in use)}
        gains = [x["delta"] for x in per.values() if x["delta"] is not None]
        scores = [x["score"] for x in per.values()]
        out.append({"entry": e, "base": base, "base_name": e["meta"].get("lora_base", ""), "modes": per,
                    "gain": sum(gains) / len(gains) if gains else None, "score": sum(scores) / len(scores) if scores else None})
    out.sort(key=lambda r: (r["gain"] is None, -(r["gain"] if r["gain"] is not None else r["score"] or 0), r["entry"]["name"].lower()))
    return out


def signed(v, d=1) -> str:
    return "–" if v is None else f"{v * 100:+.{d}f}"


# ---------------------------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------------------------
def pct(v, d=0):
    return "–" if v is None else f"{v * 100:.{d}f}%"


def dot(v):
    if v is None:
        return "⚪"
    return "🟢" if v >= 0.75 else "🟡" if v >= 0.5 else "🟠" if v >= 0.25 else "🔴"


def bar(v, n=10):
    if v is None:
        return "░" * n
    k = round(v * n)
    return "█" * k + "░" * (n - k)


def medal(i, ranked_):
    return ("🥇", "🥈", "🥉")[i] if ranked_ and i < 3 else str(i + 1)


def flag(sc) -> str:
    if not sc["complete"]:
        return f" ⚠️ <sub>{sc['n']}/{sc['of']}</sub>"
    return " <sub>partial</sub>" if sc["partial"] else ""


def params_txt(m):
    return f"{m['params_b']:g}B" if m.get("params_b") else "–"


def cell(e, key, html_sub=True):
    r = e["benchmarks"].get(key)
    if not r:
        return "–"
    sub = "" if r.get("full") else (f" <sub>n={r['n']}</sub>" if html_sub else f" (n={r['n']})")
    p5 = f" / {pct(r['pass5'], 1)}" if r.get("pass5") is not None else ""
    return f"{pct(r['value'], 1)}{p5}{sub}"


def markdown(entries: list[dict]) -> str:
    from .benchmarks import catalog as cat
    c = cat.catalog()
    now = dt.date.today().isoformat()
    L = ["<h1 align=\"center\">🏆 SLM Leaderboard</h1>", "",
         "<p align=\"center\"><i>Small language models measured locally with <b>slmbench</b> on public benchmarks via "
         f"<a href=\"https://github.com/huggingface/lighteval\">lighteval</a> · updated {now}</i></p>", ""]
    if not entries:
        return "\n".join(L + ["No models yet: run `slmbench run` and they will appear here.", ""])
    L += ["<p align=\"center\"><a href=\"leaderboard/leaderboard.html\"><b>Interactive version (HTML)</b></a> · "
          "<a href=\"docs/benchmarks.md\">Benchmarks</a></p>", "",
          "<p align=\"center\"><img src=\"leaderboard/leaderboard.svg\" alt=\"Leaderboard\" width=\"860\"></p>", ""]
    for mode, title in modes():
        rows = ranked(entries, mode)
        if not rows:
            continue
        keys = [k for k in mode_keys(mode) if any(k in e["benchmarks"] for e, _ in rows)]
        p5 = any(mode_pass5(e, mode) is not None for e, _ in rows)
        pp = mode == "long"                                  # long context: prompt processing speed matters as much
        L += [f"## {title}", "",
              "| # | Model | Params | Quant | **Score** | " + ("Score @5 | " if p5 else "") + " | ".join(c[k]["name"] for k in keys)
              + " | tok/s |" + (" prompt tok/s |" if pp else ""),
              "|:-:|---|:-:|:-:|:--|" + (":-:|" if p5 else "") + ":-:|" * len(keys) + "--:|" + ("--:|" if pp else "")]
        for i, (e, sc) in enumerate(rows):
            m, sp = e["meta"], mode_speed(e, mode)
            score = f"{dot(sc['score'])} **{pct(sc['score'])}** `{bar(sc['score'])}`" if sc["score"] is not None else "–"
            s5 = (pct(mode_pass5(e, mode)) + " | ") if p5 else ""
            L.append(f"| {medal(i, sc['ranked'])} | **{e['name']}**{flag(sc)} | {params_txt(m)} | {m.get('quant', '–')} | {score} | {s5}"
                     + " | ".join(cell(e, k) for k in keys) + f" | {sp.get('gen_tps') or '–'} |" + (f" {sp.get('prompt_tps') or '–'} |" if pp else ""))
        note = (" Cells show **pass@1 / pass@5** (n answers per problem, unbiased estimator); **Score** uses pass@1." if p5 else "")
        L += ["", f"> **Score** = mean of the {title.lower()} benchmarks (headline metric of each).{note} Medals only go to rows with every "
              "benchmark of the mode measured on the full set; the rest are listed below them: ⚠️ `k/N` = only k of the N benchmarks "
              "measured (its score averages a different set, not comparable), `partial` / `n=` = run on a sample (indicative only).", ""]
    if has_harnesses(entries):
        L += _harness_markdown(entries)
    if has_adapters(entries):
        L += _adapter_markdown(entries)
    groups: dict[str, list] = {}
    for e in entries:
        groups.setdefault(e["name"], []).append(e)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        qm = [(md, t) for md, t in modes() if any(mode_score(e, md)["n"] for v in multi.values() for e in v)]

        def qcell(e, md):
            sc = mode_score(e, md)
            return pct(sc["score"]) + ("" if sc["ranked"] or not sc["n"] else f" <sub>{sc['n']}/{sc['of']}</sub>" if not sc["complete"] else " <sub>partial</sub>")
        L += ["## Quantizations of the same model", "", "| Model | Quant | Size | " + " | ".join(t for _, t in qm) + " | tok/s |",
              "|---|:-:|:-:|" + ":-:|" * len(qm) + "--:|"]
        for name, items in multi.items():
            for e in sorted(items, key=lambda x: -(x["meta"].get("size_gb") or 0)):
                m = e["meta"]
                L.append(f"| **{name}** | {m.get('quant', '–')} | {str(m['size_gb']) + ' GB' if m.get('size_gb') else '–'} | "
                         + " | ".join(qcell(e, md) for md, _ in qm) + f" | {e['speed'].get('gen_tps') or '–'} |")
        L.append("")
    L += ["## Run details", "", "| Model | Quant | LoRA | Hardware | Backend | Size | Temp. | pass@k sampling | Reasoning budget | Max tokens | Updated |",
          "|---|:-:|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for e in sorted(entries, key=lambda x: x["name"].lower()):
        m = e["meta"]
        b = e["benchmarks"].values()
        temps = sorted({x.get("temperature") for x in b if x.get("temperature") is not None}) or ["–"]
        budgets = sorted({x.get("reasoning_budget", -1) for x in b}) or [-1]
        maxt = sorted({x.get("max_tokens") for x in b if x.get("max_tokens")}) or ["–"]
        pk = sorted({f"n={x['samples']}" + (f" @ T={x['sample_temperature']}" if x.get("sample_temperature") is not None else "")
                     if (x.get("samples") or 1) > 1 else "n=1"
                     for x in b if "samples" in x}) or ["–"]
        L.append(f"| **{e['name']}** | {m.get('quant', '–')} | {m.get('lora') or '–'} | {m.get('hardware', '–')} | {m.get('backend', '–')} | "
                 f"{str(m['size_gb']) + ' GB' if m.get('size_gb') else '–'} | {'/'.join(map(str, temps))} | {', '.join(pk)} | "
                 f"{'/'.join('unlimited' if x < 0 else str(x) for x in budgets)} | {'/'.join(map(str, maxt))} | {e['updated']} |")
    L += ["", "---", "", "<sub>Generated automatically by <code>slmbench</code> after every run. Comparisons are valid between models measured on the same "
          "machine and settings; partial runs (n=…) have large error bars.</sub>", ""]
    return "\n".join(L)


def _num(v) -> str:
    return "–" if v is None else f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def _tok(v) -> str:
    if v is None:
        return "–"
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k" if v >= 1e4 else f"{v / 1e3:.1f}k" if v >= 1e3 else f"{v:.0f}"


def _harness_markdown(entries: list[dict]) -> list[str]:
    """## Harnesses: the agent mode seen from the agents' side (which harness gets the most out of the models)."""
    from .benchmarks import catalog as cat
    c = cat.catalog()
    hk = [h for h in harness_table(entries) if h["models"]]
    L = ["## Harnesses", "",
         "SWE-bench Lite (resolved) with each agent harness: the same model can solve very different numbers of tasks "
         "depending on the agent that drives it.", "",
         "| Harness | Version | How the model acts | Models | Mean resolved | Turns/task | Tokens/task | Best model |",
         "|---|:-:|---|:-:|:--|--:|--:|---|"]
    for h in sorted(hk, key=lambda h: -(h["mean"] or 0)):
        b = h["best"]
        best = ((f"**{b['name']}** {b['quant']} · {pct(b['value'], 1)}" + ("" if b.get("full") else f" <sub>n={b['n']}</sub>"))
                if b and b["value"] else "–")
        L.append(f"| **{h['name']}** | {h['version'] or '–'} | {h['note_en']} | {h['models']} | "
                 f"{dot(h['mean'])} **{pct(h['mean'], 1)}** `{bar(h['mean'])}` | {_num(h['turns'])} | {_tok(h['tokens'])} | {best} |")
    keys = [h["key"] for h in hk]
    L += ["", "| Model | Quant | " + " | ".join(c[k].get("harness_name", k) for k in keys) + " | Best harness |",
          "|---|:-:|" + ":-:|" * len(keys) + ":-:|"]
    for e in harness_rows(entries):
        bk = (best_harness(e) or (None, 0))[0]
        cells = [("**" + cell(e, k) + "**" if k == bk else cell(e, k)) for k in keys]
        L.append(f"| **{e['name']}** | {e['meta'].get('quant', '–')} | " + " | ".join(cells)
                 + f" | {c[bk].get('harness_name', bk) if bk else '–'} |")
    L += ["", "> **Turns/task** = model calls the harness made per task; **Tokens/task** = prompt + generated tokens over them "
          "(means over the models). Resolved = the tests of the real fix pass and none breaks (official SWE-bench harness). **Mean resolved** "
          "averages the models measured with that harness; compare harnesses on the same models and task set (`n=` = a sample "
          "of the tasks; on arm64 at most 252 of the 300 have an image).", ""]
    return L


def _adapter_markdown(entries: list[dict]) -> list[str]:
    """## LoRA adapters: each fine-tune against the base model it was trained on."""
    rows = adapter_rows(entries)
    shown = [(m, t) for m, t in modes() if any(m in r["modes"] for r in rows)]
    L = ["## LoRA adapters", "",
         "Models served with a LoRA adapter, against their base model: **gain** = the adapter's score minus the base's, in "
         "points, on the benchmarks both have measured. Ranked by the mean gain over the modes.", "",
         "| # | Adapter | Base model | Quant | **Gain** | " + " | ".join(t for _, t in shown) + " |",
         "|:-:|---|---|:-:|:--|" + ":-:|" * len(shown)]
    for i, r in enumerate(rows):
        e, base = r["entry"], r["base"]
        bname = base["name"] if base else (f"{r['base_name']} <sub>not measured</sub>" if r["base_name"] else "–")
        cells = []
        for m, _t in shown:
            x = r["modes"].get(m)
            if not x:
                cells.append("–")
                continue
            sub = " <sub>partial</sub>" if x["partial"] else ""
            cells.append(f"{pct(x['score'], 1)} ({signed(x['delta'])})" + sub if x["delta"] is not None else pct(x["score"], 1) + sub)
        gain = f"**{signed(r['gain'])}** pts" if r["gain"] is not None else "–"
        L.append(f"| {i + 1} | **{e['name']}**<br><sub>{e['meta'].get('lora', '')}</sub> | {bname} | {e['meta'].get('quant', '–')} | "
                 f"{gain} | " + " | ".join(cells) + " |")
    L += ["", "> Cells: the adapter's score (gain over the base, in points). The base model is the GGUF the server was "
          "started with, or the adapter's `base_model` on Hugging Face; measure it as well (same quant) to see the gain.", ""]
    return L


def svg(entries: list[dict]) -> str:
    shown = [(m, t) for m, t in modes() if any(mode_score(e, m)["n"] for e in entries)] or modes()[:2]
    def order(e):
        sc = [mode_score(e, m) for m, _ in shown]
        vals = [x["score"] for x in sc if x["score"] is not None]
        return (-sum(x["ranked"] for x in sc), -(sum(vals) / len(vals) if vals else 0), e["name"].lower())
    rows = sorted(entries, key=order)
    W, left, right, top = 880, 250, 70, 70
    rh = max(56, 20 + 18 * len(shown))
    bw = W - left - right
    H = top + max(1, len(rows)) * rh + 30
    css = """
    .bg{fill:#ffffff}.t{fill:#1f2328;font:600 22px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
    .s{fill:#59636e;font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
    .n{fill:#1f2328;font:600 15px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
    .m{fill:#59636e;font:12px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
    .v{fill:#1f2328;font:600 12px ui-monospace,Menlo,monospace}
    .track{fill:#eaeef2}.gen{fill:#2da44e}.code{fill:#0969da}.tools{fill:#bf8700}.long{fill:#8250df}.agent{fill:#1a7f37}
    @media (prefers-color-scheme: dark){.bg{fill:#0d1117}.t,.n,.v{fill:#e6edf3}.s,.m{fill:#9198a1}.track{fill:#21262d}.gen{fill:#3fb950}.code{fill:#58a6ff}.tools{fill:#d29922}.long{fill:#a371f7}.agent{fill:#2ea043}}
    """
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="SLM leaderboard">',
         f"<style>{css}</style>", f'<rect class="bg" width="{W}" height="{H}" rx="14"/>',
         '<text class="t" x="28" y="38">🏆 SLM Leaderboard</text>', '<text class="s" x="28" y="58">public benchmarks via lighteval</text>']
    lx = W - 28 - 110 * len(shown)
    for i, (mode, lab) in enumerate(shown):
        cls = MODE_CLASS.get(mode, "gen")
        o.append(f'<rect class="{cls}" x="{lx + i * 110}" y="30" width="14" height="14" rx="3"/><text class="s" x="{lx + 20 + i * 110}" y="42">{lab}</text>')
    for i, e in enumerate(rows):
        y = top + i * rh
        m = e["meta"]
        o.append(f'<text class="n" x="28" y="{y + 18}">{escape(e["name"][:28])}</text>')
        sub = " · ".join(x for x in (params_txt(m) if m.get("params_b") else "", m.get("quant", ""),
                                     f"{e['speed'].get('gen_tps')} tok/s" if e["speed"].get("gen_tps") else "") if x)
        o.append(f'<text class="m" x="28" y="{y + 36}">{escape(sub)}</text>')
        for j, (mode, _) in enumerate(shown):
            cls = MODE_CLASS.get(mode, "gen")
            msc = mode_score(e, mode)
            v = msc["score"]
            by = y + 4 + j * 18
            o.append(f'<rect class="track" x="{left}" y="{by}" width="{bw}" height="12" rx="6"/>')
            if v is not None:
                o.append(f'<rect class="{cls}" x="{left}" y="{by}" width="{max(2, bw * v):.1f}" height="12" rx="6"/>')
            mark = "" if msc["ranked"] or v is None else "*"
            o.append(f'<text class="v" x="{left + bw + 10}" y="{by + 11}">{pct(v)}{mark}</text>')
    if any(not mode_score(e, m)["ranked"] and mode_score(e, m)["n"] for e in rows for m, _ in shown):
        o.append(f'<text class="m" x="28" y="{H - 10}">* not all benchmarks of the mode measured, or a partial (n=) run: not comparable</text>')
    o.append("</svg>")
    return "\n".join(o)


def html(entries: list[dict]) -> str:
    from .leaderboard_html import page
    return page(entries)


# ---------------------------------------------------------------------------------------------
def rebuild(s: Settings | None = None) -> dict:
    p = paths(s)
    entries = load_entries(s)
    p["dir"].mkdir(parents=True, exist_ok=True)
    p["md"].write_text(markdown(entries))
    p["html"].write_text(html(entries))
    p["svg"].write_text(svg(entries))
    return p


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def open_html(path: Path) -> None:
    """Abre `path` en el navegador. En WSL usa el navegador de Windows (xdg-open no suele tener uno configurado)."""
    path = Path(path).resolve()
    if _is_wsl():
        try:
            win = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()
            # explorer.exe devuelve código 1 aunque funcione: no se comprueba.
            subprocess.Popen(["explorer.exe", win], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (OSError, subprocess.CalledProcessError):
            pass
    import webbrowser
    if not webbrowser.open(path.as_uri()):
        raise RuntimeError(f"No se pudo abrir el navegador. Abre a mano: {path}")


def text_table(s: Settings | None = None) -> str:
    entries = load_entries(s)
    if not entries:
        return "No models in the leaderboard yet."
    L = []
    for mode, title in ((m, t.upper()) for m, t in modes()):
        rows = ranked(entries, mode)
        if not rows:
            continue
        p5 = any(mode_pass5(e, mode) is not None for e, _ in rows)
        L.append(f"{title}\n {'#':>2}  {'MODEL':<30} {'PARAMS':>6} {'QUANT':<9} {'SCORE':>6} " + (f"{'@5':>5} " if p5 else "") + f"{'MEASURED':<9} BENCHMARKS")
        for i, (e, sc) in enumerate(rows):
            m = e["meta"]
            bench = " · ".join(f"{k} {pct(e['benchmarks'][k]['value'])}"
                               + (f"/{pct(e['benchmarks'][k]['pass5'])}" if e["benchmarks"][k].get("pass5") is not None else "")
                               + ("" if e["benchmarks"][k].get("full") else f"(n={e['benchmarks'][k]['n']})")
                               for k in mode_keys(mode) if k in e["benchmarks"])
            meas = f"{sc['n']}/{sc['of']}" + ("" if sc["ranked"] else " !" if not sc["complete"] else " n=")
            L.append(f" {i + 1:>2}  {e['name'][:30]:<30} {params_txt(m):>6} {m.get('quant', '-')[:9]:<9} {pct(sc['score']):>6} "
                     + (f"{pct(mode_pass5(e, mode)):>5} " if p5 else "") + f"{meas:<9} {bench}")
        L.append("")
    return "\n".join(L)
