"""Runs INSIDE the agent sandbox with the swebench venv (/opt/swebench): grades one patch with the official SWE-bench harness
logic in a FRESH container of the task's image (no network), the same way `swebench.harness.run_evaluation` does it:
the same patch-apply fallbacks, the instance's eval script and the official log parser and report.

stdin: {"instance": <SWE-bench/SWE-bench_Lite row>, "patch": str, "image": str, "out": dir, "label": "key=value" (docker
label of the grading container), "timeout": s}
writes <out>/report.json ({instance_id: report}) and <out>/test_output.txt; prints the report as one JSON line.
"""
import json
import subprocess
import sys
from pathlib import Path

from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.utils import make_test_spec

cfg = json.loads(sys.stdin.read())
inst, patch, image = cfg["instance"], cfg.get("patch") or "", cfg["image"]
out = Path(cfg["out"])
out.mkdir(parents=True, exist_ok=True)
iid = inst["instance_id"]
spec = make_test_spec(inst)
log_path = out / "test_output.txt"
pred = {"instance_id": iid, "model_name_or_path": "slmbench", "model_patch": patch}


def sh(*argv, input_=None, timeout=600):
    return subprocess.run(argv, input=input_, capture_output=True, text=True, timeout=timeout)


if not patch.strip():
    log_path.write_text("(empty patch: nothing to grade)\n")
    report = {"patch_is_None": False, "patch_exists": False, "patch_successfully_applied": False, "resolved": False}
else:
    c = sh("docker", "run", "-d", "--network", "none", "--memory", cfg.get("memory", "4g"), "--cpus", "2", "--label", cfg["label"],
           image, "sleep", "7200").stdout.strip()
    log = []
    try:
        sh("docker", "exec", "-i", c, "sh", "-c", "cat > /tmp/patch.diff", input_=patch)
        sh("docker", "exec", "-i", c, "sh", "-c", "cat > /eval.sh", input_=spec.eval_script)
        applied = False
        for i, cmd in enumerate(GIT_APPLY_CMDS):
            if i:   # a failed attempt (notably --reject) leaves partial state behind: restart from a pristine tree
                sh("docker", "exec", "-w", "/testbed", c, "bash", "-c", "git checkout -- . ; git clean -fdq")
            r = sh("docker", "exec", "-w", "/testbed", c, "bash", "-c", f"{cmd} /tmp/patch.diff")
            if r.returncode == 0:
                applied = True
                log.append(f"{APPLY_PATCH_PASS}:\n{r.stdout}")
                break
        if not applied:
            r2 = sh("docker", "exec", "-w", "/testbed", c, "bash", "-c", "git apply --check --reverse /tmp/patch.diff")
            if r2.returncode == 0:
                applied = True
                log.append(f"{APPLY_PATCH_PASS}: verified already applied")
        if not applied:
            log.append(f"{APPLY_PATCH_FAIL}:\n{r.stdout}{r.stderr}")
        else:
            try:
                t = sh("docker", "exec", c, "/bin/bash", "/eval.sh", timeout=int(cfg.get("timeout") or 1800))
                log.append(t.stdout + t.stderr)
            except subprocess.TimeoutExpired:
                log.append(f"\n\nTimeout error: {cfg.get('timeout')} seconds exceeded.")
    finally:
        sh("docker", "rm", "-f", c)
    log_path.write_text("\n".join(log))
    report = get_eval_report(spec, pred, str(log_path), include_tests_status=True)[iid]

(out / "report.json").write_text(json.dumps({iid: report}, indent=1))
print(json.dumps({"report": report}), flush=True)
