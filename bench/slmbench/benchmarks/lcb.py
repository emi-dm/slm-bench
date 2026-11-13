"""LiveCodeBench con pass@1 y pass@5.

Usa el prompt, el dataset (lighteval/code_generation_lite, v4_v5) y el ejecutor de tests de la tarea `lcb:codegeneration` de
lighteval; solo cambia la métrica: lighteval genera una única respuesta por problema, aquí se generan las n de pass@k
(SLMBENCH_PASSK_N) y se cuentan las correctas con su mismo criterio (todas las pruebas superadas).
"""
from __future__ import annotations

import json
import re

import numpy as np
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.lcb.codegen_metrics import extract_code
from lighteval.tasks.tasks.lcb.main import lcb_codegeneration_prompt_fn

from slmbench.benchmarks.code import passk_metric

TIMEOUT = 6     # segundos por caso de prueba, como lighteval


_RUNNER_SRC: str | None = None
_DRIVER = """import json, sys
sys.path.insert(0, ".")
import lcb_runner
d = json.load(open("input.json"))
res = lcb_runner.run_test(d["sample"], test=d["code"], timeout=d["timeout"])
print("__SLM_RESULT__" + json.dumps([x.item() if hasattr(x, "item") else x for x in res]))
"""


def _runner_source() -> str:
    """lighteval's LCB test runner (the installed version), sent to the sandbox with every check."""
    global _RUNNER_SRC
    if _RUNNER_SRC is None:
        import lighteval.tasks.tasks.lcb.codegen_metrics as cm
        with open(cm.__file__, encoding="utf-8") as f:
            _RUNNER_SRC = f.read()
    return _RUNNER_SRC


def _check_in_sandbox(sample: dict, generation: str, timeout: int, n_tests: int) -> list:
    from slmbench.sandbox import execute
    files = {"lcb_runner.py": _runner_source(), "driver.py": _DRIVER,
             "input.json": json.dumps({"sample": sample, "code": generation, "timeout": timeout})}
    ok, out = execute(files, [(["python", "driver.py"], (timeout + 1) * n_tests + 5, "run")])
    m = re.search(r"__SLM_RESULT__(\[.*\])", out)
    return json.loads(m.group(1)) if m else [-1] * n_tests


def check_correctness(sample: dict, generation: str, timeout: int = TIMEOUT) -> list:
    """Per-test results of one answer, computed by lighteval's run_test *inside the slmbench sandbox* (never on the host).
    (lighteval's own check runs the code in a fork of the evaluator and, with the "spawn" start method of macOS/Windows,
    scores every answer 0; running it in the sandbox avoids both problems.)"""
    n_tests = len(json.loads(sample["input_output"])["inputs"])
    return _check_in_sandbox(sample, generation, timeout, n_tests)


def _count_passing(texts: list[str], doc) -> int:
    sample = {"input_output": json.dumps({"inputs": doc.specific["inputs"], "outputs": doc.specific["outputs"],
                                          "fn_name": doc.specific["fn_name"]})}
    def one(t):
        try:
            res = check_correctness(sample, extract_code(t))
        except Exception:                              # compilation/runner failure: wrong answer, as in lighteval (-2)
            res = [-2]
        return bool(np.all(np.array(res) > 0))       # lighteval's criterion: every test passed

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, len(texts))) as ex:   # the n answers of a problem, at the same time (all in the sandbox)
        return sum(ex.map(one, texts))


def _prompt(line: dict, task_name: str = "livecodebench"):
    return lcb_codegeneration_prompt_fn(line, task_name)


livecodebench = LightevalTaskConfig(
    name="livecodebench", prompt_function=_prompt, hf_repo="lighteval/code_generation_lite", hf_subset="v4_v5",
    hf_avail_splits=["test"], evaluation_splits=["test"], few_shots_split=None, few_shots_select=None,
    generation_size=32768, metrics=[passk_metric(_count_passing)], stop_sequence=[], version=0,
)

TASKS = [livecodebench]
