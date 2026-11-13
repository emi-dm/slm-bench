"""Failure explorer: what the model answered for each problem of a run, and why it failed.

Reads lighteval's per-problem details (prompt, answers, metrics; saved with --save-details) and joins them with slmbench's
responses.jsonl (finish reason, tokens) through the problem key (a hash of the prompt). For a single problem, `explain`
re-checks each answer to say *why* it failed: for code it runs the check again in the sandbox (compile error, wrong
answer, timeout…), for IFStruct it lists the validator's errors, for maths it shows the expected answer.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

from .benchmarks import catalog as cat
from .i18n import T


def _plain(x):
    """numpy / pyarrow values from the parquet details -> plain Python."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def problem_key(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def _answers_by_problem(run_dir: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    p = run_dir / "responses.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out.setdefault(r.get("problem", ""), {})[int(r.get("i", 0))] = r
    return out


def _status(value) -> str:
    if value is None or value != value:                       # None / NaN
        return "n/a"
    return "pass" if value >= 0.999 else "fail" if value <= 0.001 else "partial"


def load(run_dir: str | Path) -> list[dict]:
    """[{task, key, name, metric, samples: [{index, doc, metric, value, status, answers: [{i, text, reasoning, finish, ct}]}]}]"""
    run_dir = Path(run_dir)
    import pandas as pd
    answers = _answers_by_problem(run_dir)
    c = cat.catalog()
    tasks = []
    for f in sorted(glob.glob(str(run_dir / "lighteval-out" / "details" / "*" / "*" / "*.parquet"))):
        task = Path(f).name[len("details_"):].rsplit("_", 1)[0]          # e.g. "humaneval_plus|0"
        key = cat.spec_to_key(task)
        headline = c[key]["metric"] if key else None
        samples = []
        for i, row in pd.read_parquet(f).iterrows():
            doc, mr, metric = _plain(row["doc"]), _plain(row["model_response"]), _plain(row["metric"])
            msgs = mr.get("input") or []
            saved = answers.get(problem_key(msgs), {})
            texts = mr.get("text") or [""]
            reasons = mr.get("reasonings") or []
            ans = []
            for j, t in enumerate(texts):
                s = saved.get(j, {})
                ans.append({"i": j, "text": t or "", "reasoning": (reasons[j] if j < len(reasons) else None) or s.get("reasoning"),
                            "finish": s.get("finish"), "ct": s.get("ct")})
            value = metric.get(headline) if headline else next(iter(metric.values()), None)
            samples.append({"index": i, "doc": doc, "messages": msgs, "metric": metric, "value": value, "status": _status(value),
                            "answers": ans, "truncated": any(a["finish"] == "length" for a in ans),
                            "empty": any(not a["text"].strip() for a in ans)})
        tasks.append({"task": task, "key": key, "name": c[key]["name"] if key else task, "metric": headline, "samples": samples})
    return tasks


def summary_markdown(run_dir: str | Path, tasks: list[dict] | None = None) -> str:
    tasks = tasks if tasks is not None else load(run_dir)
    L = [f"# {T('Respuestas', 'Answers')}: {Path(run_dir).name}", "",
         f"| Benchmark | {T('Problemas', 'Problems')} | {T('Bien', 'Pass')} | {T('Parcial', 'Partial')} | {T('Mal', 'Fail')} | "
         f"{T('Truncadas', 'Truncated')} | {T('Vacías', 'Empty')} |", "|---|--:|--:|--:|--:|--:|--:|"]
    for t in tasks:
        sm = t["samples"]
        L.append(f"| {t['name']} | {len(sm)} | {sum(s['status'] == 'pass' for s in sm)} | {sum(s['status'] == 'partial' for s in sm)} | "
                 f"{sum(s['status'] == 'fail' for s in sm)} | {sum(s['truncated'] for s in sm)} | {sum(s['empty'] for s in sm)} |")
    if not tasks:
        L.append(T("(esta ejecución no guardó detalles por problema)", "(this run has no per-problem details)"))
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------
# why an answer failed
# ---------------------------------------------------------------------------------------------
def _classify_code(ok: bool, out: str) -> str:
    if ok:
        return T("correcta", "correct")
    o = out or ""
    if o == "timeout":
        return T("tiempo agotado", "timeout")
    if o.startswith("compile:") or "build failed" in o or "setup failed" in o:
        return T("no compila", "compile error")
    if o.startswith("sandbox unavailable"):
        return T("sandbox no disponible", "sandbox unavailable")
    return T("tests fallidos o error al ejecutar", "tests failed or runtime error")


def explain_answer(task_key: str | None, doc: dict, text: str) -> tuple[str, str]:
    """(verdict, details) for one answer. Code is re-checked in the sandbox."""
    spec = doc.get("specific") or {}
    if task_key in (None, ""):
        return "", ""
    if task_key == "lcb":
        from .benchmarks.lcb import check_correctness
        from lighteval.tasks.tasks.lcb.codegen_metrics import extract_code
        sample = {"input_output": json.dumps({"inputs": spec["inputs"], "outputs": spec["outputs"], "fn_name": spec["fn_name"]})}
        res = check_correctness(sample, extract_code(text))
        n = len(res)
        passed = sum(1 for r in res if r is True or (isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0))
        if passed == n:
            return T("correcta", "correct"), f"{n}/{n} tests"
        codes = {-1: T("tiempo agotado", "timeout"), -2: T("no compila / error de ejecución", "compile or runtime error"),
                 -3: T("tiempo agotado", "timeout"), -4: T("error de ejecución", "runtime error")}
        first_bad = next((r for r in res if r is not True and not (isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0)), None)
        why = codes.get(first_bad, T("resultado incorrecto", "wrong answer")) if not isinstance(first_bad, bool) else T("resultado incorrecto", "wrong answer")
        return why, f"{passed}/{n} tests"
    from .benchmarks import code as C
    if task_key in C.SCORERS:
        ok, out = C.SCORERS[task_key](text, spec)
        return _classify_code(ok, out), (out or "").strip()[-1500:]
    if task_key == "ifstruct":
        from .benchmarks.ifstruct import _load
        from .benchmarks.ifstruct_validator import validate_response
        keys = ("top_level_key", "require_wrapper_key", "require_code_block", "require_no_commentary", "output_format")
        v = validate_response(response=text or "", json_schema=_load(spec["json_schema"]), top_level_count=_load(spec["top_level_count"]),
                              **{k: spec[k] for k in keys})
        return (T("correcta", "correct") if v.passed else T("no cumple", "constraints not met")), "\n".join(f"- {e}" for e in v.errors[:15])
    if task_key.startswith("bfcl_"):
        from .benchmarks.bfcl import judge, parse_calls
        ok, why = judge(text, spec)
        calls = parse_calls(text)
        shown = json.dumps(calls, ensure_ascii=False, indent=1) if calls is not None else ""
        return (T("correcta", "correct") if ok else why), (f"{why}\n\n" if not ok else "") + shown
    if task_key == "mgsm_es":
        from .benchmarks.mgsm import parse_number
        got, want = parse_number(text), spec.get("answer")
        ok = got is not None and abs(got - want) < 1e-6
        return (T("correcta", "correct") if ok else T("número incorrecto", "wrong number")), T(f"esperado {want} · extraído {got}", f"expected {want} · extracted {got}")
    if task_key.startswith("ruler_"):
        refs = spec.get("answers") or []
        found = [r for r in refs if r.lower() in (text or "").lower()]
        verdict = T("correcta", "correct") if len(found) == len(refs) else T(f"{len(found)}/{len(refs)} valores encontrados", f"{len(found)}/{len(refs)} values found")
        return verdict, T(f"esperados: {', '.join(refs)}", f"expected: {', '.join(refs)}")
    return "", ""


def _reference(doc: dict) -> str:
    choices, gold = doc.get("choices") or [], doc.get("gold_index")
    gi = gold[0] if isinstance(gold, list) and gold else gold
    if isinstance(gi, int) and 0 <= gi < len(choices) and str(choices[gi]).strip():
        return str(choices[gi])
    sp = doc.get("specific") or {}
    for k in ("output", "answer", "expected", "truth"):
        if sp.get(k) not in (None, ""):
            return json.dumps(sp[k], ensure_ascii=False, indent=1) if isinstance(sp[k], (dict, list)) else str(sp[k])
    return ""


def _short(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n // 2] + f"\n… ({len(text):,} chars) …\n" + text[-n // 2:]


def sample_markdown(task: dict, sample: dict, check: bool = True) -> str:
    """Prompt, reference and every answer of one problem, with why each answer failed (check=True re-runs code checks)."""
    L = [f"# {task['name']} · {T('problema', 'problem')} {sample['index']}", "",
         f"- **{T('Resultado', 'Result')}**: {sample['status']}  ·  " + "  ·  ".join(
             f"{k} {v * 100:.0f}%" for k, v in sample["metric"].items() if isinstance(v, (int, float)) and v == v)]
    for m in sample["messages"]:
        L += ["", f"## {m.get('role', 'user')}", "", "````text", _short(str(m.get("content", "")), 2500), "````"]
    ref = _reference(sample["doc"])
    if ref:
        L += ["", f"## {T('Respuesta de referencia', 'Reference answer')}", "", "````text", _short(ref, 1200), "````"]
    for a in sample["answers"]:
        verdict, details = explain_answer(task["key"], sample["doc"], a["text"]) if check else ("", "")
        head = f"## {T('Respuesta', 'Answer')} {a['i'] + 1}/{len(sample['answers'])}"
        if verdict:
            head += f" · {verdict}"
        bits = [b for b in (f"finish: {a['finish']}" if a.get("finish") else "", f"{a['ct']} tokens" if a.get("ct") else "",
                            T("vacía", "empty") if not a["text"].strip() else "") if b]
        L += ["", head, ""] + ([f"- {'  ·  '.join(bits)}", ""] if bits else [])
        if details:
            L += [f"**{T('Por qué', 'Why')}**", "", "````text", details, "````", ""]
        if a.get("reasoning"):
            L += [f"**{T('Razonamiento', 'Reasoning')}** ({len(a['reasoning']):,} chars)", "", "````text", _short(a["reasoning"], 1200), "````", ""]
        L += ["````text", _short(a["text"], 4000) or T("(vacía)", "(empty)"), "````"]
    return "\n".join(L)
