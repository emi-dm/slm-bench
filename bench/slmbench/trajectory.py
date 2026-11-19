"""Agent trajectories (mini-SWE-agent, Pi, OpenCode on SWE-bench) for the TUI and the CLI: every step the agent took.

`find(folder)` lists the tasks of an agent run: every `*.traj.json` under the folder, joined with the SWE-bench grading
(`report.json`, `{instance_id: {resolved, tests_status…}}`) when there is one. `load(path)` turns one trajectory into
plain steps; `task_markdown`, `step_markdown` and `full_markdown` render them for the pager.

A step is one model call and what came of it:
- **reasoning**: the model's thinking (`reasoning_content`, what llama-server / vLLM return apart from the answer);
- **response**: the visible answer (mini-SWE-agent's THOUGHT + the bash block);
- **actions**: what the harness ran: bash commands (text-based configs) or tool calls with their arguments;
- **observations**: what came back: return code, output, exception (timeouts…), or the harness's own feedback when the
  answer could not be parsed (no bash block, two blocks…);
- finish reason, tokens and timings of that call (a `length` finish = the answer was cut by max_tokens).

Both mini-SWE-agent formats are read: text-based (`mswea_bash_command` blocks, `extra.actions`) and tool calling
(`tool_calls` on the assistant message, `role: tool` messages with the result).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from .i18n import T

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_OUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>", re.S)
_RC_RE = re.compile(r"<returncode>(-?\d+)</returncode>")


# ---- reading -------------------------------------------------------------------------------------------------------------

def _num(x, cast=float):
    try:
        return cast(x)
    except (TypeError, ValueError):
        return None


def _actions(msg: dict) -> list[dict]:
    """[{kind: 'bash' | 'tool', name, command, args}] of an assistant message."""
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            args = {"arguments": raw}
        cmd = args.get("command") if isinstance(args, dict) else None
        out.append({"kind": "tool", "name": fn.get("name") or "?", "command": cmd, "args": args, "id": tc.get("id")})
    if out:
        return out
    acts = (msg.get("extra") or {}).get("actions")
    if isinstance(acts, str):                     # some writers store the list as its repr
        try:
            acts = ast.literal_eval(acts)
        except (ValueError, SyntaxError):
            acts = [{"command": acts}]
    for a in acts or []:
        if isinstance(a, dict):
            out.append({"kind": "bash", "name": "bash", "command": a.get("command", ""), "args": None})
    return out


def _observation(msg: dict) -> dict:
    ex = msg.get("extra") or {}
    content = msg.get("content")
    if isinstance(content, list):                 # content parts
        content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = content or ""
    rc = _num(ex.get("returncode"), int)
    if rc is None and (m := _RC_RE.search(content)):
        rc = int(m.group(1))
    output = ex.get("raw_output")
    if output is None:
        m = _OUT_RE.search(content)
        output = m.group(1) if m else content
    return {"returncode": rc, "output": output, "exception": ex.get("exception_info") or "",
            "tool_call_id": msg.get("tool_call_id"), "feedback": rc is None and msg.get("role") != "tool",
            "raw": content}


def load(path: str | Path) -> dict:
    """{id, path, agent, exit_status, submission, stats, system, task, steps: [...], end}"""
    path = Path(path)
    if path.name.endswith((".opencode.json", ".opencode.jsonl")):
        return _load_opencode(path)
    if path.name.endswith(".jsonl"):
        return _load_pi(path)
    t = json.loads(path.read_text(encoding="utf-8"))
    info = t.get("info") or {}
    msgs = t.get("messages") or []
    system = next((m.get("content") or "" for m in msgs if m.get("role") == "system"), "")
    task, steps, end = "", [], None
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        if role == "assistant":
            ex = m.get("extra") or {}
            resp = ex.get("response") or {}
            ch = (resp.get("choices") or [{}])[0]
            rmsg = ch.get("message") or {}
            usage = resp.get("usage") or {}
            timings = resp.get("timings") or {}
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            steps.append({
                "n": len(steps) + 1,
                "reasoning": m.get("reasoning_content") or rmsg.get("reasoning_content") or rmsg.get("reasoning") or "",
                "response": content or "",
                "actions": _actions(m),
                "observations": [],
                "finish": ch.get("finish_reason"),
                "error": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "gen_tps": _num(timings.get("predicted_per_second")),
                "time": _num(ex.get("timestamp")),
            })
        elif role in ("user", "tool"):
            if not steps:                         # the task itself (instance template)
                task = m.get("content") or ""
                continue
            steps[-1]["observations"].append(_observation(m))
        elif role == "exit":
            ex = m.get("extra") or {}
            end = {"exit_status": ex.get("exit_status"), "message": m.get("content") or ""}
    prev = None
    for s in steps:                               # seconds since the previous call ended: model time + command time
        s["elapsed"] = round(s["time"] - prev, 1) if s["time"] and prev else None
        prev = s["time"] or prev
    return {
        "id": t.get("instance_id") or path.name.split(".traj")[0],
        "path": str(path),
        "agent": "mini-SWE-agent " + str(info.get("mini_version") or ""),
        "exit_status": info.get("exit_status") or (end or {}).get("exit_status") or "?",
        "submission": info.get("submission") or "",
        "stats": info.get("model_stats") or {},
        "system": system,
        "task": task,
        "steps": steps,
        "end": end,
    }


def _pi_messages(path: Path) -> list[dict]:
    """Pi's messages from any of its JSONL forms: `pi --mode json` events (message_end), a session file (type: message)
    or bare messages (role)."""
    msgs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        if e.get("type") == "message_end" or (e.get("type") == "message" and isinstance(e.get("message"), dict)):
            msgs.append(e["message"])
        elif "role" in e and "type" not in e:
            msgs.append(e)
        elif e.get("type") == "agent_settled":
            msgs.append({"role": "_settled", "aborted": bool(e.get("aborted"))})
    return msgs


def _parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [c for c in content or [] if isinstance(c, dict)]


def _load_pi(path: Path) -> dict:
    """A Pi run (https://pi.dev): assistant messages carry text, thinking and toolCall parts; each tool call is answered by
    a toolResult message with the same toolCallId."""
    msgs = _pi_messages(path)
    system = task = ""
    steps, end, last_stop = [], None, None
    for m in msgs:
        role = m.get("role")
        if role == "system":
            system = "\n".join(p.get("text", "") for p in _parts(m.get("content")) if p.get("type") == "text")
            continue
        if role == "user":
            text = "\n".join(p.get("text", "") for p in _parts(m.get("content")) if p.get("type") == "text")
            if not steps and not task:
                task = text
            elif steps:                                 # a later user turn (follow-up / steering)
                steps[-1]["observations"].append({"returncode": None, "output": text, "exception": "", "tool_call_id": None,
                                                  "feedback": True, "raw": text, "user": True})
            continue
        if role == "assistant":
            parts = _parts(m.get("content"))
            u = m.get("usage") or {}
            acts = []
            for p in parts:
                if p.get("type") == "toolCall":
                    args = p.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {"arguments": args}
                    name = str(p.get("name") or "?")
                    cmd = args.get("command") if isinstance(args, dict) and name.split("__")[-1] == "bash" else None
                    acts.append({"kind": "bash" if cmd is not None else "tool", "name": name, "command": cmd,
                                 "args": args, "id": p.get("id")})
            stop = m.get("stopReason")
            last_stop = stop
            ts = _num(m.get("timestamp"))
            steps.append({
                "n": len(steps) + 1,
                "reasoning": "\n".join(p.get("thinking", "") for p in parts if p.get("type") == "thinking"),
                "response": "\n".join(p.get("text", "") for p in parts if p.get("type") == "text"),
                "actions": acts,
                "observations": [],
                "finish": {"length": "length", "toolUse": "tool_calls", "stop": "stop"}.get(stop, stop),
                "error": m.get("errorMessage"),
                "prompt_tokens": u.get("input"),
                "completion_tokens": u.get("output"),
                "gen_tps": None,
                "time": ts / 1000 if ts and ts > 1e11 else ts,     # Pi stores milliseconds
            })
        elif role == "toolResult":
            text = "\n".join(p.get("text", "") for p in _parts(m.get("content")) if p.get("type") == "text")
            obs = {"returncode": 1 if m.get("isError") else 0, "output": text, "exception": "",
                   "tool_call_id": m.get("toolCallId"), "feedback": False, "raw": text, "tool": m.get("toolName")}
            if steps:
                steps[-1]["observations"].append(obs)
        elif role == "_settled":
            end = {"exit_status": "aborted" if m.get("aborted") else "settled", "message": ""}
    prev = None
    for s in steps:
        s["elapsed"] = round(s["time"] - prev, 1) if s["time"] and prev else None
        prev = s["time"] or prev
    ptok = sum(s["prompt_tokens"] or 0 for s in steps)
    return {
        "id": path.name.split(".pi.jsonl")[0].split(".jsonl")[0],
        "path": str(path),
        "agent": "Pi",
        "exit_status": (end or {}).get("exit_status") or (f"stop: {last_stop}" if last_stop else "?"),
        "submission": "",
        "stats": {"api_calls": len(steps), "prompt_tokens": ptok},
        "system": system,
        "task": task,
        "steps": steps,
        "end": end,
    }


def _load_opencode(path: Path) -> dict:
    """An OpenCode session: `opencode export` (messages with their parts) or, if the export is missing, the events of
    `opencode run --format json` (one part per line). Between a step-start and a step-finish part is one model call: its
    reasoning and text parts, and its tool parts (input, output, state), with the tokens and finish reason at the end."""
    task, parts = "", []
    if path.name.endswith(".json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        for m in d.get("messages") or []:
            info = m.get("info") or {}
            if info.get("role") == "user":
                if not task:
                    task = "\n".join(p.get("text", "") for p in m.get("parts") or [] if p.get("type") == "text").strip().strip('"')
                continue
            parts += [p for p in m.get("parts") or [] if isinstance(p, dict)]
            if info.get("error"):
                parts.append({"type": "_error", "error": info["error"]})
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e.get("part"), dict):
                parts.append({**e["part"], "_ts": e.get("timestamp")})
            elif e.get("type") == "error":
                parts.append({"type": "_error", "error": e.get("error") or e})
    steps: list[dict] = []
    cur = None

    def new():
        st = {"n": len(steps) + 1, "reasoning": "", "response": "", "actions": [], "observations": [], "finish": None, "error": None,
              "prompt_tokens": None, "completion_tokens": None, "gen_tps": None, "time": None}
        steps.append(st)
        return st

    for p in parts:
        t = p.get("type")
        if t == "step-start" or cur is None and t in ("reasoning", "text", "tool"):
            cur = new()
            if t == "step-start":
                continue
        if cur is None:
            continue
        end = (p.get("time") or {}).get("end") or p.get("_ts")
        if end:
            cur["time"] = end / 1000 if end > 1e11 else end
        if t == "reasoning":
            cur["reasoning"] += (p.get("text") or "")
        elif t == "text":
            cur["response"] += (p.get("text") or "")
        elif t == "tool":
            st = p.get("state") or {}
            name = str(p.get("tool") or "?")
            inp = st.get("input") or {}
            cmd = inp.get("command") if name == "bash" and isinstance(inp, dict) else None
            cur["actions"].append({"kind": "bash" if cmd is not None else "tool", "name": name, "command": cmd, "args": inp,
                                   "id": p.get("callID")})
            meta = st.get("metadata") or {}
            err = st.get("status") == "error"
            rc = meta.get("exit") if isinstance(meta.get("exit"), int) else (1 if err else 0)
            cur["observations"].append({"returncode": rc, "output": st.get("error") if err else (st.get("output") or ""), "exception": "",
                                        "tool_call_id": p.get("callID"), "feedback": False, "raw": "", "tool": name})
        elif t == "step-finish":
            tok = p.get("tokens") or {}
            reason = p.get("reason")
            cur["finish"] = {"tool-calls": "tool_calls", "max-tokens": "length", "length": "length"}.get(reason, reason)
            cur["prompt_tokens"] = tok.get("input")
            cur["completion_tokens"] = (tok.get("output") or 0) + (tok.get("reasoning") or 0) if tok else None
            cur = None
        elif t == "_error":
            err = p.get("error")
            (cur or new())["error"] = json.dumps(err, ensure_ascii=False)[:500] if not isinstance(err, str) else err
    prev = None
    for st in steps:
        st["elapsed"] = round(st["time"] - prev, 1) if st["time"] and prev else None
        prev = st["time"] or prev
    last = steps[-1]["finish"] if steps else None
    return {
        "id": path.name.split(".opencode.")[0],
        "path": str(path),
        "agent": "OpenCode",
        "exit_status": "Submitted" if last == "stop" else (f"stop: {last}" if last else "?"),
        "submission": "",
        "stats": {"api_calls": len(steps)},
        "system": "",
        "task": task,
        "steps": steps,
        "end": None,
    }


def _patches(folder: Path) -> dict:
    """{instance_id: model_patch} from the run's preds.json (SWE-bench predictions), for agents that don't keep the patch in
    the trajectory (Pi) or to show exactly what was graded."""
    out = {}
    for f in sorted(folder.rglob("preds.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        rows = d.values() if isinstance(d, dict) else d if isinstance(d, list) else []
        for r in rows:
            if isinstance(r, dict) and r.get("instance_id"):
                out[r["instance_id"]] = r.get("model_patch") or ""
    return out


def trajectory_files(folder: str | Path) -> list[Path]:
    """mini-SWE-agent (*.traj.json), Pi (*.pi.jsonl) and OpenCode (*.opencode.json, or its events *.opencode.jsonl when
    the export is missing)."""
    folder = Path(folder)
    oc = list(folder.rglob("*.opencode.json"))
    oc += [f for f in folder.rglob("*.opencode.jsonl") if not f.with_suffix(".json").exists()]
    return sorted([*folder.rglob("*.traj.json"), *folder.rglob("*.pi.jsonl"), *oc])


def _answers(folder: Path, t: dict) -> list[dict]:
    """The model's answers for this task as slmbench's relay saved them (answers.jsonl of an agent run), in call order."""
    from .benchmarks import catalog as cat
    harness = "pi" if t["agent"].startswith("Pi") else "opencode" if t["agent"].startswith("OpenCode") else "mini-swe-agent"
    keys = {k for k, v in cat.catalog().items() if v.get("harness") == harness}
    rows = []
    for f in [folder / "answers.jsonl", *folder.parents[:3]]:
        f = f if f.name == "answers.jsonl" else f / "answers.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("id") == t["id"] and r.get("key") in keys:
                    rows.append(r)
            break
    return sorted(rows, key=lambda r: r.get("n") or 0)


def _attach_rejected(t: dict, answers: list[dict]) -> None:
    """mini-SWE-agent drops the answers it can't parse and keeps only its feedback. Calls happen in order: each step is
    one answer and each feedback follows one rejected answer, so the saved answers line up with them. Attached only when
    the counts match exactly."""
    slots = []
    for s in t["steps"]:
        slots.append(None)
        slots += [o for o in s["observations"] if o["feedback"] and not o.get("user")]
    if not answers or len(answers) != len(slots):
        return
    for slot, ans in zip(slots, answers):
        if slot is not None:
            slot["rejected"] = ans


def load_in(folder: str | Path, path: str | Path) -> dict:
    """load() plus the patch from the run's preds.json when the trajectory doesn't carry one, and (agent runs) the answers
    the harness rejected."""
    t = load(path)
    if not t["submission"].strip():                   # the patch next to the trajectory (one per harness for the same task)
        own = Path(path).parent / "patch.diff"
        t["submission"] = own.read_text() if own.exists() else _patches(Path(path).parent).get(t["id"], "") or \
            _patches(Path(folder)).get(t["id"], "")
    try:
        _attach_rejected(t, _answers(Path(path).parent, t))
    except (OSError, ValueError):
        pass
    return t


def _reports(folder: Path) -> dict:
    """SWE-bench grading of the run: every report.json under the folder ({instance_id: {...}})."""
    out = {}
    for f in sorted(folder.rglob("report.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            out.update({k: v for k, v in d.items() if isinstance(v, dict) and "resolved" in v})
    return out


def find(folder: str | Path) -> list[dict]:
    """The tasks of an agent run: [{id, path, agent, steps, exit_status, resolved (None = not graded), report, truncated,
    errors, patch}]"""
    folder = Path(folder)
    reports, patches = _reports(folder), _patches(folder)
    rows = _task_rows(folder)
    tasks = []
    for f in trajectory_files(folder):
        try:
            t = load(f)
        except (OSError, ValueError) as e:
            tasks.append({"id": f.name.split(".")[0], "path": str(f), "agent": "?", "steps": 0, "exit_status": type(e).__name__,
                          "resolved": None, "report": {}, "truncated": 0, "errors": 0, "patch": False})
            continue
        rep = {}
        try:                                         # the grading next to the trajectory: one per harness for the same task
            rep = json.loads((f.parent / "report.json").read_text()).get(t["id"], {})
        except (OSError, ValueError):
            rep = reports.get(t["id"], {})
        row = rows.get((f.parent.parent.name, t["id"]), {})
        tasks.append({
            "id": t["id"], "path": str(f), "agent": t["agent"], "steps": len(t["steps"]),
            "exit_status": pretty_status(row.get("exit_status") or t["exit_status"]),
            "resolved": rep.get("resolved") if rep else None, "report": rep,
            "truncated": sum(s["finish"] == "length" for s in t["steps"]),
            "errors": sum(any(o["feedback"] and not o.get("user") for o in s["observations"]) or bool(s.get("error"))
                          for s in t["steps"]),
            "patch": bool((t["submission"] or ((f.parent / "patch.diff").read_text() if (f.parent / "patch.diff").exists()
                                               else patches.get(t["id"], ""))).strip()),
        })
    return tasks


def pretty_status(st: str) -> str:
    """'Error: {"name": "ContextOverflowError", …}' -> 'Error: ContextOverflowError'."""
    st = str(st or "?")
    if st.startswith("Error: {"):
        try:
            return "Error: " + str(json.loads(st[7:]).get("name") or "?")
        except ValueError:
            m = re.search(r'"name":\s*"([^"]+)"', st)
            return "Error: " + (m.group(1) if m else "?")
    return st


def _task_rows(folder: Path) -> dict:
    """{(harness, task id): row} from the agent run's tasks.jsonl (the real exit status: time limit, context overflow…)."""
    for d in (folder, *list(folder.parents)[:3]):
        f = d / "tasks.jsonl"
        if f.exists():
            out = {}
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                    out[(r["harness"], r["id"])] = r
                except (ValueError, KeyError):
                    continue
            return out
    return {}


def has_trajectories(folder: str | Path) -> bool:
    return bool(trajectory_files(folder))


# ---- one-line summaries ---------------------------------------------------------------------------------------------------

def _first_line(text: str, n: int = 80) -> str:
    return " ".join((text or "").split())[:n]


def task_line(t: dict, with_agent: bool = True) -> str:
    res = {True: T("✔ resuelta", "✔ resolved"), False: T("✘ no resuelta", "✘ unresolved"), None: T("· sin evaluar", "· not graded")}
    flags = []
    if t["truncated"]:
        flags.append(T(f"{t['truncated']} cortadas", f"{t['truncated']} cut"))
    if t["errors"]:
        flags.append(T(f"{t['errors']} sin comando válido", f"{t['errors']} no valid command"))
    if not t["patch"]:
        flags.append(T("sin parche", "no patch"))
    agent = f"{t.get('agent', '').split(' ')[0]:<15}" if with_agent else ""
    return (f"{t['id']:<26} {agent}{res[t['resolved']]:<15} {t['steps']:>3} " + T("pasos", "steps") + f"  {t['exit_status'][:27]:<27} "
            + " · ".join(flags))


def step_line(s: dict, last: bool = False) -> str:
    if s["actions"]:
        a = s["actions"][0]
        what = (a["command"] if a["command"] is not None else f"{a['name']}({_first_line(json.dumps(a['args'], ensure_ascii=False), 60)})")
        what = _first_line(what, 70) + (f"  (+{len(s['actions']) - 1})" if len(s["actions"]) > 1 else "")
        if SUBMIT in (a["command"] or ""):
            what = "⏎ submit"
    else:
        what = T("(respuesta final)", "(final answer)") if last and s["finish"] != "length" else T("(sin comando)", "(no command)")
    rcs = [o["returncode"] for o in s["observations"] if o["returncode"] is not None]
    rc = ("rc " + ",".join(map(str, rcs))) if rcs else ("⚠ " + T("formato", "format") if any(o["feedback"] and not o.get("user")
                                                                                          for o in s["observations"]) else "")
    if s.get("error"):
        rc = "⚠ error"
    if rcs and any(o["feedback"] and not o.get("user") for o in s["observations"]):
        rc += " ⚠"
    flags = ("✂ " if s["finish"] == "length" else "") + ("🧠 " if s["reasoning"] else "")
    tok = f"{s['completion_tokens']}t" if s["completion_tokens"] is not None else ""
    return f"#{s['n']:<3} {rc:<9} {tok:>6} {flags:<5} {what}"


# ---- Markdown -------------------------------------------------------------------------------------------------------------

def _fence(text: str, lang: str = "") -> str:
    """A code block whose fence is longer than any backtick run inside (model answers carry their own ``` blocks)."""
    text = (text or "").rstrip("\n")
    longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=2)
    f = "`" * max(3, longest + 1)
    return f"{f}{lang}\n{text}\n{f}"


def _clip(text: str, lines: int) -> str:
    parts = (text or "").split("\n")
    if lines <= 0 or len(parts) <= lines:
        return text or ""
    keep = lines // 2
    return "\n".join(parts[:keep] + [T(f"… {len(parts) - 2 * keep} líneas omitidas …", f"… {len(parts) - 2 * keep} lines omitted …")]
                     + parts[-keep:])


def step_markdown(s: dict, clip: int = 0, last: bool = False) -> str:
    """last: the run's final step (an answer without a command is how Pi ends, not a mistake)."""
    head = [T("Paso", "Step") + f" {s['n']}"]
    if s["finish"]:
        head.append(f"finish: **{s['finish']}**" + (" ✂ " + T("cortada por max_tokens", "cut by max_tokens") if s["finish"] == "length" else ""))
    if s["prompt_tokens"] is not None:
        head.append(f"{s['prompt_tokens']:,} → {s['completion_tokens']:,} tokens")
    if s["gen_tps"]:
        head.append(f"{s['gen_tps']:.0f} tok/s")
    if s["elapsed"] is not None:
        head.append(f"{s['elapsed']} s")
    md = [f"## {' · '.join(head[:1])}", "", " · ".join(head[1:]), ""]
    if s.get("error"):
        md += [f"> ⚠ {T('Error del modelo', 'Model error')}: {_first_line(s['error'], 300)}", ""]
    if s["reasoning"]:
        md += ["### 🧠 " + T("Razonamiento", "Reasoning"), "", _fence(_clip(s["reasoning"], clip)), ""]
    md += ["### 💬 " + T("Respuesta", "Response"), "", _fence(_clip(s["response"], clip)) if s["response"].strip()
           else T("(vacía)", "(empty)"), ""]
    for a in s["actions"]:
        if a["kind"] == "bash" or (a["command"] is not None and a["name"] in ("bash", "shell")):
            md += ["### $ bash" + (" · submit" if SUBMIT in (a["command"] or "") else ""), "", _fence(a["command"], "bash"), ""]
        else:
            md += [f"### 🔧 {T('Herramienta', 'Tool')}: `{a['name']}`", "",
                   _fence(json.dumps(a["args"], ensure_ascii=False, indent=2), "json"), ""]
    if not s["actions"] and not (last and s["finish"] != "length"):
        md += ["### ⚠ " + T("Ningún comando que ejecutar", "No command to run"), ""]
    for o in s["observations"]:
        if o.get("user"):
            md += ["### 👤 " + T("Mensaje del usuario", "User message"), "", _fence(_clip(o["raw"], clip)), ""]
            continue
        if o["feedback"]:
            md += ["### ↩ " + T("Aviso del harness: la siguiente respuesta del modelo no traía un comando válido",
                                "Harness feedback: the model's next answer had no valid command"), "",
                   T("mini-SWE-agent no guarda esa respuesta; esto es lo que le contestó al modelo.",
                     "mini-SWE-agent doesn't keep that answer; this is what it told the model.") if not o.get("rejected") else
                   T("La respuesta rechazada (guardada por slmbench) y lo que mini-SWE-agent le contestó al modelo.",
                     "The rejected answer (saved by slmbench) and what mini-SWE-agent told the model."), ""]
            if o.get("rejected"):
                r = o["rejected"]
                if r.get("reasoning"):
                    md += ["#### 🧠 " + T("Razonamiento", "Reasoning"), "", _fence(_clip(r["reasoning"], clip)), ""]
                md += ["#### 💬 " + T("Respuesta rechazada", "Rejected answer") + (f" · finish: {r.get('finish')}" if r.get("finish") else ""),
                       "", _fence(_clip(r.get("content") or "", clip)) if (r.get("content") or "").strip() else T("(vacía)", "(empty)"), "",
                       "#### ↩ " + T("Aviso", "Feedback"), ""]
            md += [_fence(_clip(o["raw"], clip)), ""]
            continue
        rc = o["returncode"]
        mark = "✔" if rc == 0 else "✘"
        label = f"{T('Resultado', 'Result')} · {o['tool']}" + (" · " + T("error", "error") if rc else "") if o.get("tool") \
            else f"{T('Salida', 'Output')} · rc {rc}"
        md += [f"### {mark} {label}", ""]
        if o["exception"]:
            md += [f"> ⚠ {_first_line(o['exception'], 300)}", ""]
        md += [_fence(_clip(o["output"], clip)) if (o["output"] or "").strip() else T("(sin salida)", "(no output)"), ""]
    return "\n".join(md)


def _report_md(rep: dict) -> list[str]:
    if not rep:
        return [T("Sin evaluar: no hay report.json de SWE-bench para esta tarea.", "Not graded: no SWE-bench report.json for this task."), ""]
    md = ["| " + T("Resultado", "Result") + " | |", "|---|---|",
          f"| resolved | {'✔' if rep.get('resolved') else '✘'} |",
          f"| {T('parche aplicado', 'patch applied')} | {'✔' if rep.get('patch_successfully_applied') else '✘'} |"]
    ts = rep.get("tests_status") or {}
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        d = ts.get(k) or {}
        if d:
            ok, bad = len(d.get("success") or []), len(d.get("failure") or [])
            md.append(f"| {k} | {ok}/{ok + bad} |")
    md.append("")
    bad = [(k, x) for k in ("FAIL_TO_PASS", "PASS_TO_PASS") for x in ((ts.get(k) or {}).get("failure") or [])]
    if bad:
        md += ["**" + T("Tests que fallan", "Failing tests") + "**", ""] + [f"- {k}: `{x}`" for k, x in bad[:30]] + [""]
    return md


def task_markdown(t: dict, rep: dict | None = None) -> str:
    """The task's header: grading, stats, the issue the agent received and the final patch."""
    steps = t["steps"]
    ptok = sum(s["prompt_tokens"] or 0 for s in steps)
    ctok = sum(s["completion_tokens"] or 0 for s in steps)
    md = [f"# {t['id']}", "",
          " · ".join([t.get("agent", ""), f"exit: **{t['exit_status']}**", f"{len(steps)} " + T("pasos", "steps"),
                      f"{ptok:,} " + T("tokens de entrada", "prompt tokens"), f"{ctok:,} " + T("generados", "generated"),
                      f"{sum(s['finish'] == 'length' for s in steps)} " + T("cortadas", "cut")]), ""]
    md += ["## " + T("Evaluación (SWE-bench)", "Grading (SWE-bench)"), ""] + _report_md(rep or {})
    md += ["## " + T("Parche final", "Final patch"), "",
           _fence(t["submission"], "diff") if t["submission"].strip() else T("(sin parche)", "(no patch)"), ""]
    md += ["## " + T("Tarea (lo que recibe el agente)", "Task (what the agent receives)"), "", _fence(t["task"]), ""]
    md += ["## " + T("Prompt de sistema", "System prompt"), "", _fence(t["system"]), ""]
    return "\n".join(md)


def full_markdown(t: dict, rep: dict | None = None, clip: int = 60) -> str:
    md = [task_markdown(t, rep), "---", "", "# " + T("Trayectoria", "Trajectory"), ""]
    for s in t["steps"]:
        md += [step_markdown(s, clip=clip, last=s is t["steps"][-1]), "---", ""]
    if t["end"]:
        md += [f"**exit: {t['end']['exit_status']}**", ""]
    return "\n".join(md)
