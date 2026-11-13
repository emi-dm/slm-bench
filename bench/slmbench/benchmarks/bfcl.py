"""BFCL v3 (Berkeley Function Calling Leaderboard): native tool calling, non-live single-turn Python categories.

simple (400), multiple (200), parallel (200) and parallel_multiple (200), from gorilla-llm/Berkeley-Function-Calling-Leaderboard
(Apache-2.0). The model gets the question and the functions as OpenAI `tools` and must answer with `tool_calls`, like
BFCL's "FC" mode: functions are converted as BFCL's convert_to_tool does (dots in names -> underscores, Gorilla types ->
OpenAPI types, "This is a float type value." / "Note that the provided function is in Python 3 syntax." notes). Calls are
judged with a port of BFCL's AST checker (eval_checker/ast_eval/ast_checker.py): function name, required / unexpected /
optional parameters, strict types (an int is accepted for a float, never a bool for an int), standardised strings, lists in
order, dicts, and for parallel categories an order-free greedy matching of calls to answers. Plain text instead of a tool
call is a format error (0).

The HF repo is not a loadable dataset and `datasets` alters some float answers, so `prepare()` (run by `slmbench setup`)
downloads the original files with hf_hub_download and writes a local dataset per category in which question, functions and
answers are stored as exact JSON strings.
"""
from __future__ import annotations

import copy
import json
import re

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from .bfcl_data import CATEGORIES, data_dir
from .code import passk_metric

TOOL_CALLS_PREFIX = "SLMBENCH_TOOL_CALLS:"          # how the slmbench client returns tool calls to lighteval (JSON after it)


# ---------------------------------------------------------------------------------------------
# functions -> OpenAI tools (BFCL convert_to_tool + language pre-processing)
# ---------------------------------------------------------------------------------------------
GORILLA_TO_OPENAPI = {"integer": "integer", "number": "number", "float": "number", "string": "string", "boolean": "boolean",
                      "bool": "boolean", "array": "array", "list": "array", "dict": "object", "object": "object",
                      "tuple": "array", "any": "string", "byte": "integer", "short": "integer", "long": "integer",
                      "double": "number", "char": "string", "ArrayList": "array", "Array": "array", "HashMap": "object",
                      "Hashtable": "object", "Queue": "array", "Stack": "array", "Any": "string", "String": "string",
                      "Bigint": "integer"}


def _cast(props: dict) -> dict:
    for key, value in props.items():
        if "type" not in value:
            value["type"] = "string"
        else:
            t = value["type"]
            value["type"] = GORILLA_TO_OPENAPI.get(t, "string")
            if t == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
        if value["type"] == "object" and "properties" in value:
            value["properties"] = _cast(value["properties"])
        elif value["type"] == "array" and isinstance(value.get("items"), dict):
            items = value["items"]
            it = items.get("type", "string")
            items["type"] = GORILLA_TO_OPENAPI.get(it, "string")
            if items["type"] == "object" and "properties" in items:
                items["properties"] = _cast(items["properties"])
    return props


def to_tools(functions: list[dict]) -> list[dict]:
    tools = []
    for f in copy.deepcopy(functions):
        f["name"] = re.sub(r"\.", "_", f["name"])
        f["description"] = f.get("description", "") + " Note that the provided function is in Python 3 syntax."
        params = f.get("parameters") or {"type": "object", "properties": {}}
        params["type"] = "object"
        params["properties"] = _cast(params.get("properties", {}))
        f["parameters"] = params
        tools.append({"type": "function", "function": f})
    return tools


def bfcl_prompt(line: dict, task_name: str = "bfcl") -> Doc:
    question = json.loads(line["question"])
    functions = json.loads(line["functions"])
    turn = question[0] if question and isinstance(question[0], list) else question
    user = "\n\n".join(m["content"] for m in turn if m.get("role") == "user")
    system = next((m["content"] for m in turn if m.get("role") == "system"), None)
    spec = {"tools": to_tools(functions), "functions": line["functions"], "answer": line["answer"], "category": line["category"],
            "id": line["id"]}
    if system:
        spec["system_prompt"] = system
    return Doc(task_name=task_name, query=user, choices=[line["answer"]], gold_index=0, specific=spec)


# ---------------------------------------------------------------------------------------------
# BFCL AST checker (port)
# ---------------------------------------------------------------------------------------------
PY_TYPES = {"string": str, "integer": int, "float": float, "boolean": bool, "array": list, "tuple": list, "dict": dict, "any": str}


def standardize_string(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _type_ok(value, expected: str, nested: str | None, candidates: list) -> bool:
    want = PY_TYPES.get(expected)
    if want is None:
        return True
    if expected == "float" and type(value) is int:                # int accepted for float
        return True
    if type(value) is want:
        if want is list and nested and nested in PY_TYPES:
            sub = PY_TYPES[nested]
            if all(type(v) is sub or (sub is float and type(v) is int) for v in value):
                return True
            # BFCL's type_checker applies the "variable" escape element by element: an element may also have the type of the
            # elements of an expected answer (e.g. a list of strings where the schema says list of integers)
            for cand in candidates:
                if isinstance(cand, list) and cand and all(type(v) is type(cand[0]) for v in value):
                    return True
            return False
        return True
    # "variable" escape: the expected answer itself has another type (e.g. a variable name as a string)
    real = [c for c in candidates if c != ""]
    return bool(real) and all(type(c) is type(value) for c in real)


def _std(v):
    if isinstance(v, str):
        return standardize_string(v)
    if isinstance(v, list):
        return [_std(x) for x in v]
    if isinstance(v, dict):
        return {k: _std(x) for k, x in v.items()}
    return v


def _dict_ok(value: dict, candidate: dict) -> bool:
    for k, v in value.items():
        if k not in candidate:
            return False
        allowed = candidate[k] if isinstance(candidate[k], list) else [candidate[k]]
        if _std(v) not in [_std(a) for a in allowed]:
            return False
    return all(k in value or "" in (c if isinstance(c, list) else [c]) for k, c in candidate.items())


def _value_ok(value, candidates: list) -> bool:
    if isinstance(value, str):
        return standardize_string(value) in [standardize_string(c) for c in candidates if isinstance(c, str)]
    if isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            for cand in candidates:
                if isinstance(cand, list) and len(cand) == len(value) and all(isinstance(c, dict) and _dict_ok(v, c) for v, c in zip(value, cand)):
                    return True
            return False
        return _std(value) in [_std(c) for c in candidates if isinstance(c, list)]
    if isinstance(value, dict):
        return any(isinstance(c, dict) and _dict_ok(value, c) for c in candidates)
    return value in candidates


def check_call(call: dict, function: dict, answer: dict) -> tuple[bool, str]:
    """One model call {name: args} against one function definition and its possible answer {name: {param: [allowed]}}."""
    exp_name = next(iter(answer))
    allowed = answer[exp_name]
    want = re.sub(r"\.", "_", exp_name)
    name = next(iter(call))
    if name != want:
        return False, f"wrong function: {name} (expected {want})"
    args = call[name] if isinstance(call[name], dict) else {}
    props = (function.get("parameters") or {}).get("properties", {})
    for req in (function.get("parameters") or {}).get("required", []):
        if req not in args:
            return False, f"missing required parameter: {req}"
    for p, v in args.items():
        if p not in props or p not in allowed:
            return False, f"unexpected parameter: {p}"
        cands = allowed[p]
        ptype = props[p].get("type", "any")
        nested = (props[p].get("items") or {}).get("type") if isinstance(props[p].get("items"), dict) else None
        if not _type_ok(v, ptype, nested, cands):
            return False, f"wrong type for {p}: {type(v).__name__} (expected {ptype})"
        if ptype == "float" and type(v) is int:
            v = float(v)
        if not _value_ok(v, cands):
            return False, f"wrong value for {p}: {v!r} (allowed {cands!r})"
    for p, cands in allowed.items():
        if p not in args and "" not in cands:
            return False, f"missing parameter: {p}"
    return True, "ok"


def parse_calls(text: str) -> list[dict] | None:
    """The model's tool calls as [{name: args}], or None if it answered with text."""
    text = (text or "").strip()
    if not text.startswith(TOOL_CALLS_PREFIX):
        return None
    try:
        payload = json.loads(text[len(TOOL_CALLS_PREFIX):])
    except ValueError:
        return None
    calls = []
    for c in payload.get("tool_calls", []):
        args = c.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return None
        calls.append({c.get("name", ""): args if isinstance(args, dict) else {}})
    return calls


def judge(text: str, spec: dict) -> tuple[bool, str]:
    calls = parse_calls(text)
    if calls is None:
        return False, "no tool call (answered with text)"
    functions = json.loads(spec["functions"])
    answers = json.loads(spec["answer"])
    by_name = {re.sub(r"\.", "_", f["name"]): f for f in functions}
    category = spec["category"]
    if "parallel" in category:
        if len(calls) != len(answers):
            return False, f"expected {len(answers)} calls, got {len(calls)}"
        used = set()
        for ans in answers:
            fname = re.sub(r"\.", "_", next(iter(ans)))
            hit = None
            for i, call in enumerate(calls):
                if i in used:
                    continue
                if check_call(call, by_name.get(fname, {}), ans)[0]:
                    hit = i
                    break
            if hit is None:
                return False, f"no call matches the expected {fname}(...)"
            used.add(hit)
        return True, "ok"
    if len(calls) != 1:
        return False, f"expected 1 call, got {len(calls)}"
    ans = answers[0]
    if category == "simple" and len(functions) == 1:
        # one function and one answer: judge against them directly (as BFCL's simple checker does), even if the answer
        # file names the function differently (e.g. simple_363: "restaurant_search.find_closest" vs "find_closest")
        f = functions[0]
        return check_call(calls[0], f, {f["name"]: next(iter(ans.values()))})
    fname = re.sub(r"\.", "_", next(iter(ans)))
    return check_call(calls[0], by_name.get(fname, {}), ans)


def _count_correct(texts: list[str], doc: Doc) -> int:
    """How many of a problem's n answers pass BFCL's AST check (pass@k, like the coding benchmarks)."""
    return sum(1 for t in texts if judge(t or "", doc.specific)[0])


accuracy = passk_metric(_count_correct)        # pass@1 and pass@5 from n answers per problem (SLMBENCH_PASSK_N)


def _task(cat: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"bfcl_{cat}", prompt_function=bfcl_prompt, hf_repo=str(data_dir() / cat), hf_subset="default",
        hf_avail_splits=["test"], evaluation_splits=["test"], few_shots_split=None, few_shots_select=None,
        generation_size=2048, metrics=[accuracy], stop_sequence=[], version=0,
    )


TASKS = [_task(c) for c in CATEGORIES]
