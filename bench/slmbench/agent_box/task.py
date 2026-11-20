"""Runs INSIDE the agent sandbox (system python3, stdlib only): one SWE-bench task, start to finish.

1. The agent works on the task in a container of the task's image, with no network (`docker run --network none`):
   - mini-SWE-agent (/opt/msa): its own SWE-bench runner, text-based config (one bash block per answer, no tool calls);
   - Pi (/opt/pi): print mode with JSON events, its read/write/edit/bash tools sent into the container (pi_docker.ts);
   - OpenCode (/opt/opencode): runs inside the container itself, `opencode run --format json`, session exported at the end.
   The model is reached only through the relay on 127.0.0.1 (relay.py), under /t/<instance_id>/v1.
2. The patch (the agent's `git diff`) is graded by grade.py with the official SWE-bench harness, in a fresh container.

stdin: one JSON object (see `slmbench.agent._task_config`). stdout: JSON lines: {"ev": "phase" | "step" | "done", ...}.
Everything the task produced is left in cfg["dir"]: trajectory (*.traj.json / *.pi.jsonl / *.opencode.json), preds.json, patch.diff,
report.json, test_output.txt, task.log.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

cfg = json.loads(sys.stdin.readline())
iid, harness, image = cfg["instance"]["instance_id"], cfg["harness"], cfg["image"]
wd = Path(cfg["dir"])
wd.mkdir(parents=True, exist_ok=True)
label = cfg["label"]
log = open(wd / "task.log", "a", buffering=1)
t0 = time.time()


def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def sh(*argv, input_=None, timeout=600):
    return subprocess.run(argv, input=input_, capture_output=True, text=True, timeout=timeout)


def kill_group(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except OSError:
            return
        try:
            p.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass


def remove_containers():
    ids = sh("docker", "ps", "-aq", "--filter", f"label=slmbench-task={label}-{iid}").stdout.split()
    if ids:
        sh("docker", "rm", "-f", *ids)


def sampling() -> dict:
    return {k: v for k, v in (cfg.get("extra_body") or {}).items()}


# ---------------------------------------------------------------------------------------------
# mini-SWE-agent
# ---------------------------------------------------------------------------------------------
# SWE-bench images activate the task's conda env (testbed: the repo's Python, pytest…) in .bashrc, which a login shell
# reads (mini-SWE-agent runs `bash -lc`) but `docker exec … bash -c` doesn't (Pi's and OpenCode's tools): without this
# their `python` is conda's base one, with no pytest and none of the repo's dependencies.
TESTBED_ENV = ["-e", "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/condabin:/opt/miniconda3/bin:/usr/local/sbin:"
               "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "-e", "CONDA_DEFAULT_ENV=testbed",
               "-e", "CONDA_PREFIX=/opt/miniconda3/envs/testbed"]
# a tool call the model wrote as text (its chat template's markers) instead of making it: the harness takes it as the
# final answer and stops
TEXT_TOOL_CALL = re.compile(r"<\s*\|?\s*tool_call|<tool_call>|\[TOOL_CALLS\]|<function=|<\|python_tag\|>")


def run_mini() -> tuple[str, str]:
    data = wd / "data"
    data.mkdir(exist_ok=True)
    row = {**cfg["instance"], "image_name": image}
    (data / "test.jsonl").write_text(json.dumps(row) + "\n")
    conf = {
        "environment": {"run_args": ["--rm", "--network", "none", "--memory", cfg.get("memory", "4g"), "--cpus", "2",
                                     "--label", f"slmbench-task={label}-{iid}"],
                        "timeout": int(cfg.get("command_timeout", 120))},
        "model": {"model_name": "openai/slmbench", "model_class": "litellm_textbased", "cost_tracking": "ignore_errors",
                  "model_kwargs": {"api_base": cfg["api_base"], "api_key": "none", "temperature": cfg["temperature"],
                                   "max_tokens": cfg["max_tokens"], "drop_params": True,
                                   **({"extra_body": sampling()} if sampling() else {})}},
        "agent": {"step_limit": int(cfg["step_limit"]), "cost_limit": 0},
    }
    (wd / "mini.yaml").write_text(json.dumps(conf, indent=1))      # JSON is YAML
    out = wd / "mini-out"
    home = wd / "home"                                   # its own HOME and caches: nothing shared between tasks or runs
    home.mkdir(exist_ok=True)
    env = dict(os.environ, HOME=str(home), HF_HOME=str(home / "hf"), XDG_CACHE_HOME=str(home / ".cache"),
               XDG_CONFIG_HOME=str(home / ".config"), HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1",
               MSWEA_COST_TRACKING="ignore_errors", MSWEA_SILENT_STARTUP="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    cmd = ["/opt/msa/bin/mini-extra", "swebench", "--subset", str(data), "--split", "test", "-c", "swebench_backticks.yaml",
           "-c", str(wd / "mini.yaml"), "-o", str(out), "-w", "1"]
    log.write("$ " + " ".join(cmd) + "\n")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=str(wd))
    status = None
    try:
        p.wait(timeout=cfg["timeout"])
    except subprocess.TimeoutExpired:
        status = "TimeLimit"
        kill_group(p)
    remove_containers()
    traj = out / iid / f"{iid}.traj.json"
    patch = ""
    try:
        patch = json.loads((out / "preds.json").read_text())[iid].get("model_patch") or ""
    except (OSError, ValueError, KeyError):
        pass
    if traj.exists():
        shutil.copy(traj, wd / f"{iid}.traj.json")
        if status is None:
            try:
                status = json.loads(traj.read_text())["info"].get("exit_status")
            except (OSError, ValueError, KeyError):
                pass
    return status or ("Error" if p.returncode else "?"), patch


# ---------------------------------------------------------------------------------------------
# Pi
# ---------------------------------------------------------------------------------------------
PI_PROMPT = """<issue>
{issue}
</issue>

The repository with this issue is checked out in /testbed, your working directory. Make the changes to the non-test source
files that resolve the issue, in a way that is general and consistent with the codebase. Do not modify the tests or the
configuration files (setup.py, pyproject.toml, setup.cfg, tox.ini…). The Python environment of the project is already active
in bash; there is no network access, so do not try to install anything.

Recommended workflow: find and read the relevant code, write a small script that reproduces the issue, fix the source code,
run your script again to confirm the fix and think about edge cases. When you are done, answer with a short summary of the
change and no tool call: your changes to the files in /testbed are the result."""


def run_pi() -> tuple[str, str]:
    c = sh("docker", "run", "-d", "--network", "none", "--memory", cfg.get("memory", "4g"), "--cpus", "2", "-w", "/testbed", *TESTBED_ENV,
           "--label", f"slmbench-task={label}-{iid}", image, "sleep", str(int(cfg["timeout"]) + 600)).stdout.strip()
    if not c:
        return "Error: container did not start", ""
    agent_dir = wd / "pi-agent"
    agent_dir.mkdir(exist_ok=True)
    models = {"providers": {"slmbench": {
        "name": "slmbench", "baseUrl": cfg["api_base"], "api": "openai-completions", "apiKey": "none",
        "models": [{"id": "slmbench", "name": cfg.get("model", "slmbench"), "reasoning": False, "input": ["text"],
                    "contextWindow": int(cfg.get("context") or 32768), "maxTokens": int(cfg["max_tokens"]),
                    "samplingParams": {"temperature": cfg["temperature"], **sampling()}}]}}}
    (agent_dir / "models.json").write_text(json.dumps(models, indent=1))
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(agent_dir), PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1", PI_TELEMETRY="0",
               SLMBENCH_CONTAINER=c, PATH="/opt/node/bin:" + os.environ.get("PATH", ""))
    cmd = ["/opt/pi/node_modules/.bin/pi", "-p", "--mode", "json", "--no-session", "--no-context-files", "--no-skills",
           "--no-prompt-templates", "--tools", "read,bash,edit,write", "-e", cfg["pi_extension"], "--model", "slmbench/slmbench",
           PI_PROMPT.format(issue=cfg["instance"]["problem_statement"])]
    log.write("$ pi -p --mode json … (" + c[:12] + ")\n")
    events = open(wd / f"{iid}.pi.jsonl", "w", buffering=1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd="/testbed", start_new_session=True,
                         bufsize=1)
    status, steps = None, 0
    deadline = time.time() + float(cfg["timeout"])
    timer = threading.Timer(float(cfg["timeout"]), lambda: kill_group(p))
    timer.start()
    try:
        for line in p.stdout:
            events.write(line)
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "message_end" and (e.get("message") or {}).get("role") == "assistant":
                steps += 1
                if steps >= int(cfg["step_limit"]):
                    status = "LimitsExceeded"
                    kill_group(p)
                    break
            elif e.get("type") == "agent_settled":
                status = "Aborted" if e.get("aborted") else "Submitted"
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        kill_group(p)
    finally:
        timer.cancel()
        events.close()
    if status is None:
        status = "TimeLimit" if time.time() >= deadline else ("Submitted" if p.returncode == 0 else f"Error (exit {p.returncode})")
    r = sh("docker", "exec", "-w", "/testbed", c, "bash", "-c",
           "git config --global --add safe.directory /testbed >/dev/null 2>&1; git add -A >/dev/null 2>&1 && git diff --cached")
    patch = r.stdout if r.returncode == 0 else ""
    remove_containers()
    return status, patch


# ---------------------------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------------------------
def run_opencode() -> tuple[str, str]:
    """OpenCode's tools run where OpenCode runs and can't be redirected, so OpenCode itself runs INSIDE the task container
    (its binary is copied in). The container shares the sandbox's network namespace (--network host) to reach the relay on
    127.0.0.1; the sandbox has no network, so that is all it can reach."""
    c = sh("docker", "run", "-d", "--network", "host", "--memory", cfg.get("memory", "4g"), "--cpus", "2", "-w", "/testbed", *TESTBED_ENV,
           "--label", f"slmbench-task={label}-{iid}", image, "sleep", str(int(cfg["timeout"]) + 600)).stdout.strip()
    if not c:
        return "Error: container did not start", ""
    r = sh("docker", "cp", cfg["opencode"], f"{c}:/usr/local/bin/opencode")
    rg = str(Path(cfg["opencode"]).with_name("rg"))           # OpenCode's grep/glob/list tools run ripgrep
    if r.returncode == 0 and Path(rg).exists():
        r = sh("docker", "cp", rg, f"{c}:/usr/local/bin/rg")
    if r.returncode != 0:
        remove_containers()
        return f"Error: could not copy opencode: {r.stderr.strip()[:200]}", ""
    conf = {
        "$schema": "https://opencode.ai/config.json", "model": "slm/slmbench", "small_model": "slm/slmbench",
        "provider": {"slm": {"npm": "@ai-sdk/openai-compatible", "name": "slmbench",
                             "options": {"baseURL": cfg["api_base"], "apiKey": "none"},
                             "models": {"slmbench": {"name": cfg.get("model", "slmbench"),
                                                     "limit": {"context": int(cfg.get("context") or 32768), "output": int(cfg["max_tokens"])}}}}},
        "agent": {"build": {"temperature": cfg["temperature"], "steps": int(cfg["step_limit"])}},
        # no questions to a user who isn't there; nothing that needs the network
        "permission": {"*": "allow", "webfetch": "deny", "websearch": "deny", "question": "deny"},
        "share": "disabled", "autoupdate": False, "lsp": False, "formatter": False,
    }
    env = {"OPENCODE_CONFIG_CONTENT": json.dumps(conf), "HOME": "/tmp/oc-home", "XDG_CONFIG_HOME": "/tmp/oc-home/.config",
           "XDG_DATA_HOME": "/tmp/oc-home/.local/share", "XDG_CACHE_HOME": "/tmp/oc-home/.cache", "XDG_STATE_HOME": "/tmp/oc-home/.state",
           "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
           "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1", "OPENCODE_DISABLE_CLAUDE_CODE": "1", "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
           "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1"}
    envargs = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
    cmd = ["docker", "exec", "-w", "/testbed", *envargs, c, "opencode", "run", "--format", "json", "--title", iid,
           PI_PROMPT.format(issue=cfg["instance"]["problem_statement"])]
    log.write(f"$ opencode run --format json … ({c[:12]})\n")
    events = open(wd / f"{iid}.opencode.jsonl", "w", buffering=1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True, bufsize=1)
    status, steps, session, last_text, last_reason = None, 0, None, "", ""
    deadline = time.time() + float(cfg["timeout"])
    timer = threading.Timer(float(cfg["timeout"]), lambda: kill_group(p))
    timer.start()
    try:
        for line in p.stdout:
            events.write(line)
            try:
                e = json.loads(line)
            except ValueError:
                continue
            session = session or e.get("sessionID")
            part = e.get("part") or {}
            if e.get("type") == "text":
                last_text = part.get("text") or ""
            if e.get("type") == "step_finish":
                steps += 1
                last_reason = part.get("reason") or ""
                if steps >= int(cfg["step_limit"]):
                    status = "LimitsExceeded"
                    kill_group(p)
                    sh("docker", "exec", c, "pkill", "-f", "opencode run")
                    break
            elif e.get("type") == "error":
                err = e.get("error") or {}
                status = "Error: " + (str(err.get("name") or "") if isinstance(err, dict) and err.get("name") else json.dumps(err)[:200])
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        kill_group(p)
    finally:
        timer.cancel()
        events.close()
    if status is None:
        status = "TimeLimit" if time.time() >= deadline else ("Submitted" if p.returncode == 0 else f"Error (exit {p.returncode})")
    if status == "Submitted" and last_reason == "stop" and TEXT_TOOL_CALL.search(last_text):
        status = "ToolCallAsText"                     # it didn't finish: its last tool call came as text and OpenCode stopped
    if session:                                       # the whole session (reasoning, every tool call) for the viewer
        x = sh("docker", "exec", *envargs, c, "opencode", "export", session, timeout=120)
        if x.returncode == 0 and x.stdout.strip().startswith("{"):
            (wd / f"{iid}.opencode.json").write_text(x.stdout)
    r = sh("docker", "exec", "-w", "/testbed", c, "bash", "-c",
           "git config --global --add safe.directory /testbed >/dev/null 2>&1; git add -A >/dev/null 2>&1 && git diff --cached")
    patch = r.stdout if r.returncode == 0 else ""
    remove_containers()
    return status, patch


RUNNERS = {"mini-swe-agent": lambda: run_mini(), "pi": lambda: run_pi(), "opencode": lambda: run_opencode()}

# ---------------------------------------------------------------------------------------------
if harness == "regrade":                               # rescore: grade the patch of an earlier run again, no model
    emit(ev="phase", phase="grading")
    status, patch = cfg.get("exit_status") or "?", cfg.get("patch") or ""
else:
    emit(ev="phase", phase="agent")
    try:
        status, patch = RUNNERS[harness]()
    except Exception as e:  # noqa: BLE001 - one task never stops the run
        status, patch = f"Error: {type(e).__name__}: {e}", ""
        remove_containers()
    (wd / "patch.diff").write_text(patch)
    (wd / "preds.json").write_text(json.dumps({iid: {"instance_id": iid, "model_name_or_path": cfg.get("model", ""),
                                                     "model_patch": patch}}, indent=1))
    emit(ev="phase", phase="grading")
agent_time = round(time.time() - t0, 1)
report = {}
try:
    g = subprocess.run(["/opt/swebench/bin/python", "-I", cfg["grader"]], capture_output=True, text=True, timeout=3600,
                       input=json.dumps({"instance": cfg["instance"], "patch": patch, "image": image, "out": str(wd),
                                         "label": f"slmbench-task={label}-{iid}", "timeout": cfg.get("test_timeout", 1800),
                                         "memory": cfg.get("memory", "4g")}))
    log.write(g.stderr[-4000:])
    report = json.loads(g.stdout.strip().splitlines()[-1])["report"] if g.returncode == 0 else {"error": g.stderr[-500:]}
except Exception as e:  # noqa: BLE001
    report = {"error": f"{type(e).__name__}: {e}"}
remove_containers()
emit(ev="done", exit_status=status, patch=bool(patch.strip()), resolved=bool(report.get("resolved")), report=report,
     agent_time=agent_time, total_time=round(time.time() - t0, 1))
