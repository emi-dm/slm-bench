"""Fichas de los benchmarks con una tarea real de muestra (lo que recibe el modelo), para la TUI, el CLI y docs/benchmarks.md.

La muestra se obtiene con la API pública de Hugging Face (datasets-server, primera fila del split de evaluación) y se pasa por la
misma `prompt_function` que usa lighteval, así que se ve exactamente el prompt que recibirá el modelo. Se cachea en
~/.cache/slmbench/samples/ para poder verla sin conexión después.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from ..i18n import T, get_lang
from . import catalog as cat

TASKS_FILE = Path(__file__).resolve().parent.parent / "lighteval_tasks.py"
_registry_cache: dict = {}


def _cache_dir() -> Path:
    d = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "slmbench" / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def task_config(key: str):
    """LightevalTaskConfig del benchmark (de lighteval o de las tareas propias)."""
    spec = cat.catalog()[key]["spec"]
    if spec in _registry_cache:
        return _registry_cache[spec]
    from lighteval.tasks.registry import Registry
    reg = Registry(tasks=spec, custom_tasks=str(TASKS_FILE), load_multilingual=False)
    cfgs = next(iter(reg.task_to_configs.values()))
    _registry_cache[spec] = cfgs[0]
    return cfgs[0]


def sample_row(key: str) -> dict | None:
    f = _cache_dir() / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except ValueError:
            pass
    cfg = task_config(key)
    split = list(cfg.evaluation_splits)[0]
    row = None
    try:                                    # API de vista previa de HF: rápida, pero recorta las celdas grandes
        q = urllib.parse.urlencode({"dataset": cfg.hf_repo, "config": cfg.hf_subset or "default", "split": split})
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/first-rows?{q}", timeout=30) as r:
            rows = json.load(r).get("rows") or []
        if rows and not rows[0].get("truncated_cells"):
            row = rows[0]["row"]
    except Exception:
        row = None
    if row is None:                         # celdas recortadas (p. ej. LiveCodeBench) o API caída: primera fila con `datasets`
        import datasets
        ds = datasets.load_dataset(cfg.hf_repo, cfg.hf_subset, split=split, streaming=True)
        row = next(iter(ds))
    f.write_text(json.dumps(row, ensure_ascii=False, default=str))
    return row


def _short(text: str, n: int = 1600) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n // 2] + f"\n… ({len(text):,} chars) …\n" + text[-n // 3:]


def sample(key: str) -> dict:
    """{system, prompt, reference} de la primera tarea del benchmark."""
    cfg = task_config(key)
    row = sample_row(key)
    if row is None:
        raise RuntimeError("empty dataset")
    doc = cfg.prompt_function(row, cfg.name)
    sp = doc.specific if isinstance(doc.specific, dict) else {}
    ref = ""
    if doc.choices and isinstance(doc.gold_index, int) and 0 <= doc.gold_index < len(doc.choices) and str(doc.choices[doc.gold_index]).strip():
        ref = str(doc.choices[doc.gold_index])
    for k in ("truth", "reference", "output", "expected"):
        if not ref and sp.get(k) is not None:
            ref = json.dumps(sp[k], ensure_ascii=False, indent=1) if isinstance(sp[k], (dict, list)) else str(sp[k])
    return {"system": sp.get("system_prompt"), "prompt": doc.query, "reference": ref}


def card(key: str, sample_: bool = True, head: bool = True) -> str:
    """The benchmark's card as Markdown. head=False: only the sample task (the TUI draws its own header)."""
    v = cat.catalog()[key]
    lang = get_lang()
    L = [] if not head else [f"# {v['name']}  ·  `{key}`", "",
         f"- **{T('Modo', 'Mode')}**: {v['mode']}  ·  **{T('Métrica', 'Metric')}**: `{v['metric']}`  ·  **{T('Tareas', 'Tasks')}**: {v['n']:,}",
         f"- **Dataset**: `{v['hf']}`  ·  **lighteval**: `{v['spec']}`" + ("" if v.get("builtin") else T(" (tarea propia de slmbench)", " (slmbench custom task)")),
         f"- **{T('Referencia', 'Reference')}**: {v['link']}"]
    if head and v.get("requires"):
        miss = cat.missing_tools(key)
        L.append(f"- **{T('Requiere', 'Requires')}**: {', '.join(v['requires'])}" + (f"  ⚠ {T('falta', 'missing')}: {', '.join(miss)}" if miss else "  ✔"))
    if head:
        L += ["", v.get(f"desc_{lang}") or v.get("desc_es", ""), ""]
    if sample_:
        if head:
            L += [f"## {T('Tarea de muestra (lo que recibe el modelo)', 'Sample task (what the model receives)')}", ""]
        try:
            smp = sample(key)
            if smp["system"]:
                L += [f"**system**", "", "````text", _short(smp["system"], 900), "````", ""]
            L += [f"**user**", "", "````text", _short(smp["prompt"]), "````", ""]
            if smp["reference"]:
                L += [f"**{T('Respuesta de referencia', 'Reference answer')}**", "", "````text", _short(smp["reference"], 700), "````", ""]
        except Exception as e:  # noqa: BLE001
            L += [T(f"(No se pudo obtener la muestra: {type(e).__name__}. Hace falta conexión la primera vez.)",
                    f"(Could not fetch the sample: {type(e).__name__}. A connection is needed the first time.)"), ""]
    return "\n".join(L)


def overview(mode: str | None = None) -> str:
    c = cat.catalog()
    lang = get_lang()
    L = [f"# {T('Benchmarks de slmbench', 'slmbench benchmarks')}", "",
         T("Todos son benchmarks públicos y se ejecutan con lighteval. Ficha y muestra: `slmbench benchmarks <clave>`.",
           "All are public benchmarks run through lighteval. Card and sample: `slmbench benchmarks <key>`."), ""]
    for m, info in cat.MODES.items():
        if mode and mode not in (m, "all"):
            continue
        L += [f"## {T('Modo', 'Mode')} `{m}` — {info.get(lang, m)}", "", info.get(f"desc_{lang}", ""), "",
              f"| {T('Clave', 'Key')} | Benchmark | {T('Tareas', 'Tasks')} | {T('Métrica', 'Metric')} | {T('Requiere', 'Requires')} |", "|---|---|--:|---|---|"]
        for k, v in c.items():
            if v["mode"] != m:
                continue
            miss = cat.missing_tools(k)
            req = ", ".join(v.get("requires") or []) or "—"
            L.append(f"| `{k}` | {v['name']} | {v['n']:,} | `{v['metric']}` | {req}{' ⚠' if miss else ''} |")
        L.append("")
    return "\n".join(L)


def markdown_catalog() -> str:
    """docs/benchmarks.md (in English): per-mode summary and the full card of every benchmark with its sample task."""
    from .. import i18n
    prev = i18n.get_lang()
    i18n.set_lang("en")
    try:
        return _markdown_catalog()
    finally:
        i18n.set_lang(prev)


def _markdown_catalog() -> str:
    c = cat.catalog()
    L = ["# Benchmarks used", "",
         "slmbench only uses **existing public benchmarks**, run with [lighteval](https://github.com/huggingface/lighteval) against your server. "
         "The ones that ship with lighteval are used as they are; the rest have been adapted as custom tasks (see [adapting-a-benchmark.md](adapting-a-benchmark.md)). "
         "This document is generated with `slmbench benchmarks --markdown > docs/benchmarks.md`.", ""]
    for m, info in cat.MODES.items():
        L += [f"## Mode `{m}` — {info['en']}", "", info["desc_en"], "", "| Benchmark | What it measures | Dataset | Tasks | Metric | Origin |", "|---|---|---|--:|---|---|"]
        for k, v in c.items():
            if v["mode"] == m:
                L.append(f"| [{v['name']}](#{k}) | {v['desc_en']} | `{v['hf']}` | {v['n']:,} | `{v['metric']}` | {'lighteval' if v.get('builtin') else 'custom task'} |")
        L.append("")
    L += ["## Validation and quality notes", "",
          "Before using any model, every adapted benchmark was validated by scoring its **reference answers** (they should score ~100%). "
          "All the code checks run inside the slmbench sandbox (a Docker Sandboxes microVM with no network and no access to your files) and "
          "were re-validated there: the same pass counts as below, and 22/22 MultiPL-E hand-written controls.", "",
          "| Benchmark | Reference answers that pass | Note |", "|---|--:|---|",
          "| HumanEval+ | 163/164 | The HF test for HumanEval/32 (`find_zero`) calls `_poly` with its arguments swapped; slmbench fixes that test. Even so, the reference solution fails 2 of 886 ill-conditioned inputs |",
          "| MBPP+ | 377/378 | Task 590 differs only in the last decimal digit (precision of the stored value) |",
          "| CRUXEval-O / CRUXEval-I | 800/800 · 800/800 | — |",
          "| MultiPL-E | — | The dataset has no reference solutions: it was validated with hand-written solutions (correct and incorrect) in all 5 languages and by compiling all 791 problems. Three do not compile because of defects in the dataset, so nobody can pass them: JS HumanEval_51, C++ HumanEval_137 and Go HumanEval_87 |",
          "| IFStruct | — (no reference answers) | Scored with the official validator from `Liquid4All/ifstruct`, vendored unchanged (its 20 upstream tests pass). The prompt is sent alone as the user message, like the official client, and the 2,000 HF rows match the official `data/test.jsonl` exactly (prompts and specs). End to end, on 100 prompts against the same llama-server, slmbench and a client sending exactly the official request gave the **same pass rate (11.0% vs 11.0%)**, identical answers on 89/100 and the same pass/fail on 96/100 (2 differences each way). Those differences come from llama.cpp's nondeterminism under concurrent load: the same client flips pass/fail between calls on 2 of them, and sent alone slmbench's exact request reproduces the official-style answer |",
          "| LiveCodeBench | correct / wrong / timeout controls behave as expected | lighteval's prompt, dataset and tests (`run_test`); slmbench adds pass@1/pass@5 over n answers. **lighteval's own LCB checker scores every answer 0 on macOS** (it starts a process with a nested function, which the default `spawn` start method can't pickle), even a correct one: slmbench uses a fixed copy with the same per-test results as lighteval's under `fork` (verified on a correct, a wrong and a timing-out solution to abc366_a, all 21 tests). Answers get at least 8,192 tokens (with 2,048 every answer was truncated); small reasoning models benefit from `--max-tokens 16384` |",
          "| BFCL (4 categories) | 1000/1000 · broken answers rejected 1000/1000 | AST checker ported from BFCL (eval_checker/ast_eval/ast_checker.py): calls built from every official answer, with parallel calls shuffled, all pass; changing one argument value makes every one fail. Two quirks of the HF v3 data are handled like the official checker: simple_363 names the function differently in questions and answers, and some answers allow optional parameters that aren't in the schema |",
          "| MGSM (es) | 250/250 · wrong number accepted 0/250 | Official criterion (last number of the answer), also with thousands separators |",
          "| RULER NIAH 4k / 8k | 4000/4000 · 4000/4000 | string_match_all as in RULER; an answer missing one value always scores below 100% |",
          "", "pass@k (coding): n answers per problem (default 5, sampled at temperature 0.6) and the unbiased estimator of Chen et al. (2021), identical to lighteval's `estimate_pass_at_k`. "
          "With 5 copies of the reference solution a problem scores pass@1 = pass@5 = 1; with 2 correct out of 5, pass@1 = 0.4 and pass@5 = 1.", "",
          "The lighteval benchmarks (IFEval, GSM8K, MATH-500) use lighteval's implementation unchanged.", "",
          "## Benchmark cards", ""]
    for k in c:
        body = card(k)
        L += [f'<a id="{k}"></a>', "", body.replace("# ", "### ", 1), "", "---", ""]
    return "\n".join(L)
