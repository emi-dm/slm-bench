"""Export runs as a dataset (JSONL or Parquet): one row per attempt, with the task, the whole session, the model's solution
and whether it is correct.

Every mode uses the same columns, so runs of any mode can go into one file:

| column | what |
|---|---|
| `run`, `date`, `model`, `quant`, `label` | which run and model (quant as the server reported it) |
| `mode`, `benchmark`, `benchmark_name`, `harness`, `harness_version` | what was measured (harness only in the agent mode) |
| `task_id`, `repo`, `sample` | the problem (SWE-bench instance id, or the problem's index) and the answer's number (pass@k) |
| `task` | what the model was asked: the GitHub issue, or the user prompt |
| `messages` | the session, in chat format: `role` (system, user, assistant, tool), `content`, `reasoning`, `tool_calls` (JSON), `tool_call_id`, `name`, `rejected` (an answer the harness refused: no valid command) |
| `solution` | the model's solution: the patch (`git diff`) in the agent mode, the answer otherwise |
| `correct`, `score`, `metric` | the verdict: SWE-bench `resolved`, or the benchmark's metric for the problem (`score`). With several answers per problem (pass@k) lighteval scores the problem, not each answer: `correct` is true/false when all its answers agree (score 1 or 0) and empty when only some are right |
| `exit_status`, `steps`, `finish`, `prompt_tokens`, `completion_tokens`, `truncated`, `seconds` | how it went |
| `details` | JSON with the rest: tests (FAIL_TO_PASS / PASS_TO_PASS), patch applied, generation settings |
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .benchmarks import catalog as cat
from .i18n import T

MESSAGE_FIELDS = ("role", "content", "reasoning", "tool_calls", "tool_call_id", "name", "rejected")


def _msg(role: str, content: str = "", reasoning: str = "", tool_calls=None, tool_call_id: str = "", name: str = "",
         rejected: bool = False) -> dict:
    return {"role": role, "content": content or "", "reasoning": reasoning or "",
            "tool_calls": json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "",
            "tool_call_id": tool_call_id or "", "name": name or "", "rejected": bool(rejected)}


def _meta(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "summary.json").read_text())["meta"]
    except (OSError, ValueError, KeyError):
        try:
            info = json.loads((run_dir / "run.json").read_text())
            return {"model": info.get("model"), "specs": info.get("specs"), "settings": info.get("settings", {})}
        except (OSError, ValueError):
            return {}


def _base(run_dir: Path, meta: dict) -> dict:
    return {"run": run_dir.name, "date": meta.get("date") or "", "model": meta.get("model") or "",
            "quant": (meta.get("server") or {}).get("quant") or "", "label": meta.get("label") or ""}


# ---------------------------------------------------------------------------------------------
# agent mode: one row per task and harness
# ---------------------------------------------------------------------------------------------
def _agent_messages(t: dict) -> list[dict]:
    """The trajectory as the session the model lived: system, the task, then for every step the answer (with its
    reasoning and tool calls) and what came back. Answers a harness rejected are kept where they happened."""
    out = []
    if t.get("system"):
        out.append(_msg("system", t["system"]))
    if t.get("task"):
        out.append(_msg("user", t["task"]))
    for s in t["steps"]:
        calls = [{"id": a.get("id") or "", "type": "function", "function": {"name": a["name"], "arguments": json.dumps(a["args"], ensure_ascii=False)}}
                 for a in s["actions"] if a["kind"] == "tool" or a.get("id")]
        out.append(_msg("assistant", s["response"], s["reasoning"], calls or None))
        for o in s["observations"]:
            if o.get("rejected"):
                r = o["rejected"]
                out.append(_msg("assistant", r.get("content") or "", r.get("reasoning") or "", rejected=True))
            if o.get("tool_call_id") or o.get("tool"):
                out.append(_msg("tool", o["output"] or "", tool_call_id=o.get("tool_call_id") or "", name=o.get("tool") or ""))
            else:
                out.append(_msg("user", o["raw"] or o["output"] or ""))
    return out


def agent_rows(run_dir: Path) -> list[dict]:
    from . import agent, trajectory
    meta = _meta(run_dir)
    hv = meta.get("harness_versions") or {}
    tasks = {}
    for line in (run_dir / "tasks.jsonl").read_text().splitlines() if (run_dir / "tasks.jsonl").exists() else []:
        try:
            r = json.loads(line)
            tasks[(r["harness"], r["id"])] = r
        except (ValueError, KeyError):
            continue
    try:
        problems = {i["instance_id"]: i for i in agent.instances()}
    except Exception:  # noqa: BLE001 - the dataset isn't needed for the rows, only for the issue text
        problems = {}
    c = cat.catalog()
    rows = []
    for f in trajectory.trajectory_files(run_dir / "agent"):
        harness = f.parent.parent.name
        t = trajectory.load_in(f.parent, f)
        r = tasks.get((harness, t["id"]), {})
        key = cat.spec_to_key(r.get("spec", "")) or next((k for k, v in c.items() if v.get("harness") == harness), "")
        rep = {}
        try:
            rep = json.loads((f.parent / "report.json").read_text()).get(t["id"], {})
        except (OSError, ValueError):
            pass
        patch = (f.parent / "patch.diff").read_text() if (f.parent / "patch.diff").exists() else t["submission"]
        inst = problems.get(t["id"], {})
        rows.append({
            **_base(run_dir, meta), "mode": "agent", "benchmark": key, "benchmark_name": c.get(key, {}).get("name", key),
            "harness": harness, "harness_version": hv.get(harness, ""), "task_id": t["id"], "repo": r.get("repo") or inst.get("repo", ""),
            "sample": 0, "task": inst.get("problem_statement") or t["task"], "messages": _agent_messages(t), "solution": patch,
            "correct": bool(r.get("resolved", rep.get("resolved", False))), "score": 1.0 if r.get("resolved", rep.get("resolved")) else 0.0,
            "metric": "resolved", "exit_status": str(r.get("exit_status") or t["exit_status"]), "steps": len(t["steps"]),
            "finish": t["steps"][-1]["finish"] or "" if t["steps"] else "",
            "prompt_tokens": sum(s["prompt_tokens"] or 0 for s in t["steps"]),
            "completion_tokens": sum(s["completion_tokens"] or 0 for s in t["steps"]),
            "truncated": sum(s["finish"] == "length" for s in t["steps"]), "seconds": float(r.get("agent_time") or 0),
            "details": json.dumps({"tests_status": rep.get("tests_status"), "patch_applied": rep.get("patch_successfully_applied"),
                                   "f2p": r.get("f2p"), "p2p_broken": r.get("p2p_broken"),
                                   "settings": {k: meta.get(k) for k in ("temperature", "max_new_tokens", "reasoning_budget",
                                                                         "step_limit", "task_timeout")},
                                   "dataset": meta.get("dataset")}, ensure_ascii=False),
        })
    return rows


# ---------------------------------------------------------------------------------------------
# the other modes: one row per answer (pass@k: n answers per problem)
# ---------------------------------------------------------------------------------------------
def _text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in x)
    return "" if x is None else str(x)


TOOL_CALLS_PREFIX = "SLMBENCH_TOOL_CALLS:"           # how slmbench's client hands tool calls to lighteval (bfcl.py)


def _answer(text: str) -> tuple[str, list | None]:
    """(content, tool_calls in OpenAI format) of a saved answer: tool-calling answers travel through lighteval as
    'SLMBENCH_TOOL_CALLS:{"tool_calls": [...], "content": ...}'."""
    if not (text or "").startswith(TOOL_CALLS_PREFIX):
        return text or "", None
    try:
        d = json.loads(text[len(TOOL_CALLS_PREFIX):])
    except ValueError:
        return text, None
    calls = [{"id": c.get("id") or "", "type": "function",
              "function": {"name": c.get("name") or (c.get("function") or {}).get("name", ""),
                           "arguments": c.get("arguments") if "arguments" in c else (c.get("function") or {}).get("arguments", "")}}
             for c in d.get("tool_calls") or [] if isinstance(c, dict)]
    return d.get("content") or "", calls


def lighteval_rows(run_dir: Path) -> list[dict]:
    from . import review
    meta = _meta(run_dir)
    rows = []
    for task in review.load(run_dir):
        key = task["key"] or task["task"]
        mode = cat.catalog().get(task["key"] or "", {}).get("mode", "")
        for smp in task["samples"]:
            msgs = smp["messages"] or []
            prompt = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
            for a in smp["answers"]:
                session = [_msg(m.get("role", "user"), _text(m.get("content"))) for m in msgs]
                content, calls = _answer(a["text"])
                session.append(_msg("assistant", content, a.get("reasoning") or "", calls))
                v = smp["value"]
                score = float(v) if isinstance(v, (int, float)) and v == v else None
                rows.append({
                    **_base(run_dir, meta), "mode": mode, "benchmark": key, "benchmark_name": task["name"], "harness": "",
                    "harness_version": "", "task_id": str(smp["index"]), "repo": "", "sample": int(a["i"]), "task": prompt,
                    "messages": session,
                    "solution": json.dumps([c["function"] for c in calls], ensure_ascii=False) if calls is not None else content,
                    # the metric is the problem's: with several answers (pass@k) it says which one is right only when all agree
                    "correct": True if smp["status"] == "pass" else False if smp["status"] == "fail" else None, "score": score,
                    "metric": task["metric"] or "", "exit_status": "", "steps": 1, "finish": a.get("finish") or "",
                    "prompt_tokens": 0, "completion_tokens": int(a.get("ct") or 0), "truncated": int(a.get("finish") == "length"),
                    "seconds": 0.0,
                    "details": json.dumps({"metrics": smp["metric"], "doc": smp["doc"],
                                           "note": "the problem's metric (pass@k: over all its answers)"}, ensure_ascii=False, default=str),
                })
    return rows


# ---------------------------------------------------------------------------------------------
def rows(run_dirs: list[str | Path], on_line=print) -> list[dict]:
    out = []
    for d in run_dirs:
        d = Path(d)
        try:
            got = agent_rows(d) if (d / "agent").is_dir() else lighteval_rows(d)
        except Exception as e:  # noqa: BLE001 - one unreadable run doesn't stop the export
            on_line(f"⚠ {d.name}: {type(e).__name__}: {e}")
            continue
        on_line(f"  {d.name}: {len(got)} " + T("filas", "rows"))
        out += got
    return out


def write(data: list[dict], path: str | Path, fmt: str) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        msg = pa.struct([(k, pa.bool_() if k == "rejected" else pa.string()) for k in MESSAGE_FIELDS])
        schema = pa.schema([
            ("run", pa.string()), ("date", pa.string()), ("model", pa.string()), ("quant", pa.string()), ("label", pa.string()),
            ("mode", pa.string()), ("benchmark", pa.string()), ("benchmark_name", pa.string()), ("harness", pa.string()),
            ("harness_version", pa.string()), ("task_id", pa.string()), ("repo", pa.string()), ("sample", pa.int32()),
            ("task", pa.string()), ("messages", pa.list_(msg)), ("solution", pa.string()), ("correct", pa.bool_()),
            ("score", pa.float64()), ("metric", pa.string()), ("exit_status", pa.string()), ("steps", pa.int32()),
            ("finish", pa.string()), ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()), ("truncated", pa.int32()),
            ("seconds", pa.float64()), ("details", pa.string()),
        ])
        pq.write_table(pa.Table.from_pylist(data, schema=schema), path, compression="zstd")
    return path


def default_path(out_dir: str, fmt: str, what: str = "runs") -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(out_dir).expanduser() / "exports" / f"slmbench-{what}-{stamp}.{'jsonl' if fmt == 'jsonl' else 'parquet'}"


def export(run_dirs: list[str | Path], fmt: str = "jsonl", path: str | Path | None = None, out_dir: str = ".",
           on_line=print) -> tuple[Path, int]:
    data = rows(run_dirs, on_line=on_line)
    if not data:
        raise RuntimeError(T("Nada que exportar: esas ejecuciones no tienen respuestas guardadas.",
                             "Nothing to export: those runs have no saved answers."))
    what = "agent" if all(r["mode"] == "agent" for r in data) else "runs"
    p = write(data, path or default_path(out_dir, fmt, what), fmt)
    return p, len(data)
