"""Agent mode: the model as a coding agent on SWE-bench Lite, driven by mini-SWE-agent, Pi or OpenCode, all
inside sandboxes.

What a task is: a real GitHub issue of a Python project (django, sympy, requests, pytest…) and the repository at the commit
before its fix. The agent explores the repository, edits the code and runs commands until it decides it is done; its patch
(`git diff`) is then graded with the official SWE-bench harness: the task counts as **resolved** when the tests that the real
fix makes pass now pass (FAIL_TO_PASS) and no test that passed before breaks (PASS_TO_PASS).

Where everything runs (no model output ever touches the host):

    host: slmbench ── relay thread ──────────────▶ your model server (OpenAI API)
            │  ▲ sbx exec -i (stdin/stdout)
            ▼  │
    sbx  slmbench-run-<pid>-a<hex>   ← created from the `slmbench-agent` template with --deny-network '**': no network at all
       ├─ relay.py on 127.0.0.1 (the agents' only way to the model: chat completions only, model fixed by slmbench)
       ├─ mini-SWE-agent (/opt/msa) or Pi (/opt/pi + Node), one process per task
       ├─ docker: one container per task from the task's image, --network none; every command of the agent runs there
       └─ grading (/opt/swebench): the patch in a FRESH container of the image, official eval script, log parser and report
    sbx  slmbench-run-<pid>-f<hex>   ← fetches the task images (registry only); they reach the run sandbox through a pipe
                                       (docker save | docker load), so the run sandbox never needs the network

Both sandboxes are deleted when the run ends (a sandbox left by a process that died is removed by the next run).
`slmbench setup --mode agent` builds the template once (mini-SWE-agent, swebench, Node + Pi) and downloads the dataset.

Images: the official SWE-bench images are x86_64. On arm64 (Apple Silicon, Linux arm64) slmbench uses Epoch AI's arm64
builds of the same environments (ghcr.io/epoch-research/swe-bench.eval.arm64.*), available for 252 of the 300 Lite tasks;
the other 48 (matplotlib, scikit-learn, xarray, astropy) are skipped and the report says so.
"""
from __future__ import annotations

import atexit
import base64
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import secrets
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import sandbox
from .benchmarks import catalog as cat
from .servers import effective_key
from .config import Settings
from .i18n import T

MINI_SWE_AGENT = "2.4.6"
SWEBENCH = "5.0.2"
NODE = "22.20.0"                                   # Pi needs Node.js ≥ 22.19
OPENCODE = "1.19.5"
RIPGREP = "14.1.1"                                 # OpenCode's grep/glob/list tools need `rg`; it can't download it there
DATASET = "SWE-bench/SWE-bench_Lite"
VERSION = "4"                                      # bump to rebuild the template when the tools above change
TEMPLATE = f"slmbench-agent:v{VERSION}"
BUILDER = "slmbench-agent-build"
MARKER = "/etc/slmbench-agent"
BOX = "/home/agent/slm"                            # slmbench's files inside the run sandbox
HERE = Path(__file__).resolve().parent / "agent_box"
ARM64_REGISTRY = "ghcr.io/epoch-research/swe-bench.eval.arm64."
MAX_PARALLEL = 3                                   # the sandbox's Docker disk (~10 GB) holds about this many task images
TASK_MEMORY = 4                                    # GB per task container (the agent's, then the grading one: never both)
BASE_MEMORY = 2                                    # GB for the rest of the run sandbox: relay, mini-SWE-agent / Pi, Docker
FETCH_MEMORY = "2g"                                # the fetcher only pulls and streams images


def run_memory(parallel: int) -> str:
    """The run sandbox's memory cap (sbx would give it half of the host's RAM): what its tasks can use, and no more."""
    return f"{BASE_MEMORY + TASK_MEMORY * max(1, parallel)}g"


def harnesses() -> dict[str, str]:
    """benchmark key -> harness name"""
    return {k: v["harness"] for k, v in cat.catalog().items() if v.get("mode") == "agent"}


def is_agent_spec(spec: str) -> bool:
    k = cat.spec_to_key(spec)
    return bool(k and cat.catalog()[k].get("mode") == "agent")


def pi_version() -> str:
    from .browser import PI
    return PI[1]


# ---------------------------------------------------------------------------------------------
# Data: SWE-bench Lite (test split) and the image of each task
# ---------------------------------------------------------------------------------------------
def _dir() -> Path:
    from .engine import data_dir
    d = data_dir() / "swebench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def _http(url: str, *, method: str = "GET", headers: dict | None = None, timeout: float = 60):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "slmbench", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def download_dataset(on_line=print) -> bool:
    """SWE-bench Lite test split (the current dataset, with the eval script and log parser of each task), pinned to the
    revision downloaded."""
    try:
        with _http(f"https://huggingface.co/api/datasets/{DATASET}") as r:
            rev = json.loads(r.read())["sha"]
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{rev}/data/test-00000-of-00001.parquet"
        with _http(url, timeout=300) as r:
            data = r.read()
    except (OSError, ValueError, KeyError) as e:
        on_line(f"✘ {DATASET}: {e}")
        return False
    (_dir() / "lite_test.parquet").write_bytes(data)
    (_dir() / "dataset.json").write_text(json.dumps({"dataset": DATASET, "revision": rev, "sha256": hashlib.sha256(data).hexdigest()}))
    on_line(f"✔ {DATASET} @ {rev[:10]} ({len(data) / 1e6:.1f} MB)")
    return True


def dataset_info() -> dict:
    try:
        return json.loads((_dir() / "dataset.json").read_text())
    except (OSError, ValueError):
        return {}


def instances() -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(_dir() / "lite_test.parquet")
    return [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()} for row in df.to_dict("records")]


def _ghcr_has(image: str) -> bool:
    repo, _, tag = image.removeprefix("ghcr.io/").partition(":")
    try:
        with _http(f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io", timeout=30) as r:
            token = json.loads(r.read())["token"]
        accept = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                            "application/vnd.docker.distribution.manifest.v2+json",
                            "application/vnd.docker.distribution.manifest.list.v2+json"])
        with _http(f"https://ghcr.io/v2/{repo}/manifests/{tag or 'latest'}", method="HEAD", timeout=30,
                   headers={"Authorization": f"Bearer {token}", "Accept": accept}) as r:
            return r.status == 200
    except (OSError, ValueError, KeyError):
        return False


def image_for(inst: dict, arch_: str | None = None) -> str:
    if (arch_ or arch()) == "arm64":
        return f"{ARM64_REGISTRY}{inst['instance_id']}:latest"
    return inst.get("image") or f"swebench/sweb.eval.x86_64.{inst['instance_id'].replace('__', '_1776_')}:latest".lower()


def check_images(on_line=print) -> dict:
    """{instance_id: image} of the tasks that have an image for this machine's architecture (saved for the runs)."""
    a = arch()
    insts = instances()
    if a == "arm64":
        on_line(T(f"  comprobando qué tareas tienen imagen arm64 ({len(insts)})…", f"  checking which tasks have an arm64 image ({len(insts)})…"))
        with ThreadPoolExecutor(16) as ex:
            ok = list(ex.map(lambda i: _ghcr_has(image_for(i, a)), insts))
        avail = {i["instance_id"]: image_for(i, a) for i, good in zip(insts, ok) if good}
    else:
        avail = {i["instance_id"]: image_for(i, a) for i in insts}
    (_dir() / f"images-{a}.json").write_text(json.dumps(avail, indent=0))
    on_line(f"✔ {len(avail)}/{len(insts)} " + T(f"tareas con imagen para {a}", f"tasks with an image for {a}")
            + (" (Epoch AI arm64)" if a == "arm64" else ""))
    return avail


def available() -> dict:
    try:
        return json.loads((_dir() / f"images-{arch()}.json").read_text())
    except (OSError, ValueError):
        return {}


def _order(iid: str) -> str:
    """A fixed order that mixes the repositories (the dataset is sorted by repo): the first N tasks are the same on every
    machine and every run, so partial runs are comparable."""
    return hashlib.sha256(iid.encode()).hexdigest()


def select(max_samples: int) -> list[dict]:
    avail = available()
    pool = sorted((i for i in instances() if i["instance_id"] in avail), key=lambda i: _order(i["instance_id"]))
    return pool[:max_samples] if max_samples else pool


# ---------------------------------------------------------------------------------------------
# The template: mini-SWE-agent, swebench (grading) and Node + Pi, inside a sandbox image
# ---------------------------------------------------------------------------------------------
def template_ready() -> bool:
    return any(sandbox._same_ref(t, TEMPLATE) for t in sandbox._templates())


_status: dict = {}


def status(refresh: bool = False) -> str | None:
    """None if agent runs can start; otherwise what is missing. Cached for a minute (the TUI asks on every screen)."""
    if not refresh and _status and time.time() - _status["t"] < 60:
        return _status["v"]
    _status.update(t=time.time(), v=_status_now())
    return _status["v"]


def _status_now() -> str | None:
    if not sandbox.sbx_installed():
        return "Docker Sandboxes (sbx) not installed"
    if not sandbox.daemon_ok():
        return "sbx daemon not running"
    if not template_ready():
        return "agent sandbox not set up (slmbench setup --mode agent)"
    if not (_dir() / "lite_test.parquet").exists() or not available():
        return "SWE-bench Lite not downloaded (slmbench setup --mode agent)"
    return None


_PROVISION = r"""
set -e
export UV_PYTHON_INSTALL_DIR=/opt/python UV_CACHE_DIR=/tmp/uv-cache DEBIAN_FRONTEND=noninteractive
mkdir -p /opt /testbed && chown agent:agent /testbed
command -v uv >/dev/null || {{ curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null; }}
echo "· mini-SWE-agent {mini}"
uv venv -q -p 3.12 /opt/msa && uv pip install -q -p /opt/msa/bin/python "mini-swe-agent=={mini}"
echo "· swebench {swebench} (grading)"
uv venv -q -p 3.12 /opt/swebench && uv pip install -q -p /opt/swebench/bin/python "swebench=={swebench}"
echo "· Node.js {node} + Pi {pi}"
A=$(uname -m | sed 's/aarch64/arm64/; s/x86_64/x64/')
rm -rf /opt/node && curl -fsSL "https://nodejs.org/dist/v{node}/node-v{node}-linux-$A.tar.gz" | tar -xz -C /opt && mv "/opt/node-v{node}-linux-$A" /opt/node
mkdir -p /opt/pi && cd /opt/pi && PATH=/opt/node/bin:$PATH npm install --no-fund --no-audit --save-exact "@earendil-works/pi-coding-agent@{pi}" >/tmp/npm.log 2>&1 || {{ tail -5 /tmp/npm.log; exit 1; }}
echo "· OpenCode {opencode} (runs inside the task containers)"
mkdir -p /opt/opencode/bin && cd /opt/opencode && PATH=/opt/node/bin:$PATH npm install --no-fund --no-audit --save-exact "opencode-ai@{opencode}" >/tmp/npm.log 2>&1 || {{ tail -5 /tmp/npm.log; exit 1; }}
OC=$(find /opt/opencode/node_modules -path "*/opencode-linux-*/bin/opencode" -type f ! -path "*musl*" ! -path "*baseline*" | head -1)
[ -n "$OC" ] && cp "$OC" /opt/opencode/bin/opencode && chmod 755 /opt/opencode/bin/opencode && /opt/opencode/bin/opencode --version >/dev/null
echo "· ripgrep {rg} (for OpenCode's search tools)"
case $(uname -m) in aarch64) RT=aarch64-unknown-linux-gnu ;; *) RT=x86_64-unknown-linux-musl ;; esac
curl -fsSL "https://github.com/BurntSushi/ripgrep/releases/download/{rg}/ripgrep-{rg}-$RT.tar.gz" | tar -xz -C /tmp && cp "/tmp/ripgrep-{rg}-$RT/rg" /opt/opencode/bin/rg && rm -rf "/tmp/ripgrep-{rg}-$RT" && /opt/opencode/bin/rg --version >/dev/null
chmod -R a+rX /opt
rm -rf /tmp/uv-cache /tmp/npm.log /home/agent/.cache /home/agent/.config/mini-swe-agent
chown -R agent:agent /home/agent
HOME=/tmp/check /opt/msa/bin/python -c "import minisweagent" && HOME=/tmp/check /opt/swebench/bin/python -c "import swebench.harness.grading" && HOME=/tmp/check /opt/node/bin/node /opt/pi/node_modules/.bin/pi --version >/dev/null && rm -rf /tmp/check
docker info >/dev/null
echo {version} > {marker}
"""


def provision(on_line=print) -> bool:
    """Builds the agent template: a sandbox with the tools (from PyPI, nodejs.org and npm, allowed by the sandbox's
    default policy only while it is built), saved with `sbx template save` and then removed."""
    if not sandbox.sbx_installed():
        on_line("✗ Docker Sandboxes (`sbx`) is not installed: https://docs.docker.com/ai/sandboxes/get-started/")
        return False
    if not sandbox.daemon_ok():
        on_line("✗ The Docker Sandboxes daemon is not answering: run `sbx daemon start` (or `sbx diagnose`)")
        return False
    sandbox.prune()
    if template_ready():
        on_line(T(f"✔ plantilla {TEMPLATE} lista", f"✔ template {TEMPLATE} ready"))
        return True
    name = BUILDER
    if sandbox.exists(name):
        sandbox._sbx("rm", "--force", name, timeout=300)
    on_line(T(f"  creando el sandbox {name} (sin carpetas del host)…", f"  creating sandbox {name} (no host folders)…"))
    with _LIVE_LOCK:
        _LIVE.add(name)
    r = sandbox._sbx("create", "--no-progress", "--name", name, "shell", timeout=900)
    if r.returncode != 0:
        on_line("✗ " + (r.stderr or r.stdout).strip()[-400:])
        _forget(name)
        return False
    try:
        script = _PROVISION.format(mini=MINI_SWE_AGENT, swebench=SWEBENCH, node=NODE, pi=pi_version(), opencode=OPENCODE, rg=RIPGREP,
                                   version=VERSION, marker=MARKER)
        on_line(T("  instalando las herramientas dentro del sandbox (unos minutos)…", "  installing the tools inside the sandbox (a few minutes)…"))
        p = subprocess.Popen(["sbx", "--no-input", "exec", "-u", "root", name, "bash", "-c", script], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=sandbox._sbx_env())
        for line in p.stdout:
            if line.strip():
                on_line("    " + line.rstrip())
        if p.wait() != 0:
            on_line(T("✗ la instalación falló", "✗ the installation failed"))
            return False
        on_line(T(f"  guardando la plantilla {TEMPLATE}…", f"  saving the template {TEMPLATE}…"))
        sandbox._sbx("stop", name, timeout=300)
        r = sandbox._sbx("template", "save", name, TEMPLATE, timeout=3600)
        if r.returncode != 0 or not template_ready():
            on_line("✗ could not save the template: " + (r.stderr or r.stdout).strip()[-400:])
            return False
    finally:
        _forget(name)
    for old in sandbox._templates():
        if any(sandbox._same_ref(old, f"slmbench-agent:v{v}") for v in range(1, int(VERSION))):
            sandbox._sbx("template", "rm", "--force", old, timeout=300)
    on_line(T(f"✔ plantilla {TEMPLATE}: mini-SWE-agent {MINI_SWE_AGENT}, Pi {pi_version()}, OpenCode {OPENCODE}, swebench {SWEBENCH}",
              f"✔ template {TEMPLATE}: mini-SWE-agent {MINI_SWE_AGENT}, Pi {pi_version()}, OpenCode {OPENCODE}, swebench {SWEBENCH}"))
    return True


def setup(on_line=print, download: bool = True) -> dict:
    rep = {"template": provision(on_line=on_line), "dataset": False, "available": 0}
    if download or not (_dir() / "lite_test.parquet").exists():
        rep["dataset"] = download_dataset(on_line=on_line)
    else:
        rep["dataset"] = (_dir() / "lite_test.parquet").exists()
    if rep["dataset"]:
        rep["available"] = len(check_images(on_line=on_line))
    return rep


# ---------------------------------------------------------------------------------------------
# Sandboxes of a run
# ---------------------------------------------------------------------------------------------
_LIVE: set[str] = set()                           # sandboxes this process created and hasn't removed yet
_LIVE_LOCK = threading.Lock()


def _forget(name: str) -> None:
    sandbox.destroy(name)
    with _LIVE_LOCK:
        _LIVE.discard(name)


@atexit.register
def _cleanup() -> None:
    """Whatever way slmbench ends (also SIGTERM / SIGHUP, which the CLI turns into a normal exit), its sandboxes go with it.
    Only a SIGKILL skips this; the next run or TUI start removes those (sandbox.prune: their owner is gone)."""
    with _LIVE_LOCK:
        left = list(_LIVE)
    for n in left:
        _forget(n)


def _create(kind: str, template: str | None, memory: str) -> str:
    """kind 'a': the run sandbox (from the template, no network); 'f': the image fetcher (registry access). Both with an
    explicit memory cap and half of the host's cores (at most 8)."""
    name = f"{sandbox.PREFIX}{os.getpid()}-{kind}{secrets.token_hex(3)}"
    args = ["create", "--no-progress", "--name", name, "--memory", memory, "--cpus", str(sandbox.default_cpus())]
    if template:
        args += ["--pull", "never", "-t", template, "--deny-network", "**"]
    with _LIVE_LOCK:
        _LIVE.add(name)
    try:
        r = sandbox._sbx(*args, "shell", timeout=900)
    except BaseException:
        _forget(name)
        raise
    if r.returncode != 0:
        _forget(name)
        raise RuntimeError(f"could not create sandbox {name}: " + (r.stderr or r.stdout).strip()[-300:])
    return name


def _upload(box: str, files: dict[str, bytes], dest: str) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for n, data in files.items():
            ti = tarfile.TarInfo(n)
            ti.size, ti.mode = len(data), 0o644
            tar.addfile(ti, io.BytesIO(data))
    r = subprocess.run(["sbx", "--no-input", "exec", "-i", box, "sh", "-c", f"mkdir -p {dest} && tar -xf - -C {dest}"],
                       input=buf.getvalue(), capture_output=True, env=sandbox._sbx_env(), timeout=120)
    if r.returncode != 0:
        raise RuntimeError("could not copy slmbench's files into the sandbox: " + r.stderr.decode(errors="replace")[-300:])


class Images:
    """Task images: pulled in the fetcher sandbox, piped into the run sandbox (docker save | docker load), removed from the
    fetcher at once and from the run sandbox when the task is graded."""

    def __init__(self, run_box: str, fetch_box: str):
        self.run_box, self.fetch_box = run_box, fetch_box
        self.lock = threading.Lock()                       # one pull at a time: bandwidth and the fetcher's disk

    def _x(self, box, *argv, timeout=1800, **kw):
        return subprocess.run(["sbx", "--no-input", "exec", box, *argv], capture_output=True, text=True, timeout=timeout,
                              env=sandbox._sbx_env(), **kw)

    def fetch(self, image: str) -> None:
        with self.lock:
            last = ""
            for _ in range(3):
                r = self._x(self.fetch_box, "docker", "pull", "-q", image)
                if r.returncode == 0:
                    break
                last = (r.stderr or r.stdout).strip()[-300:]
                time.sleep(5)
            else:
                raise RuntimeError(f"docker pull {image}: {last}")
            env = sandbox._sbx_env()
            save = subprocess.Popen(["sbx", "--no-input", "exec", self.fetch_box, "docker", "save", image], stdout=subprocess.PIPE, env=env)
            load = subprocess.run(["sbx", "--no-input", "exec", "-i", self.run_box, "docker", "load", "-q"], stdin=save.stdout,
                                  capture_output=True, text=True, env=env, timeout=1800)
            save.stdout.close()
            save.wait(timeout=60)
            self._x(self.fetch_box, "docker", "rmi", "-f", image, timeout=300)
            if load.returncode != 0 or save.returncode != 0:
                raise RuntimeError(f"docker load {image}: " + (load.stderr or "").strip()[-300:])

    def remove(self, image: str) -> None:
        try:
            self._x(self.run_box, "docker", "rmi", "-f", image, timeout=300)
        except subprocess.SubprocessError:
            pass


# ---------------------------------------------------------------------------------------------
# Relay: the sandbox's only way to the model
# ---------------------------------------------------------------------------------------------
_KEYLIKE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|(?i:api[_-]?key)[\"':= ]+[A-Za-z0-9_\-.]{12,})")


class _NoCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """The API key goes only to the server it was configured for: dropped if the server redirects to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        import urllib.parse
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoCrossHostAuth)
_BASH_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)```", re.S)
_PATH_RE = re.compile(r"^/t/([^/]+)/v1/chat/completions/?$")


class Relay:
    """Host side of relay.py. Only POST /t/<task>/v1/chat/completions passes, to the model under test (the request's model
    is replaced by it, and slmbench's reasoning budget / extra body are added); /v1/models answers locally; anything else is
    refused. Every call is recorded in requests.jsonl (speed, tokens, finish) and reported as a step of its task."""

    def __init__(self, box: str, s: Settings, model: str, stats: Path, on_step, server_kind: str):
        self.box, self.s, self.model, self.stats, self.on_step = box, s, model, stats, on_step
        self.extra = {}
        if s.reasoning_budget >= 0 and server_kind != "api":
            self.extra["thinking_budget_tokens"] = int(s.reasoning_budget)
        try:
            self.extra.update(json.loads(s.extra_body or "{}"))
        except ValueError:
            pass
        self.lock, self.slock = threading.Lock(), threading.Lock()
        self.max_tokens = None
        self.steps: dict[str, int] = {}
        self.usage: dict[str, dict] = {}                  # per task: turns (model calls) and tokens, for tasks.jsonl
        self.errors: list[str] = []
        self.port = random.randint(20000, 60000)
        self.p = None

    def start(self) -> None:
        self.p = subprocess.Popen(["sbx", "--no-input", "exec", "-i", self.box, "python3", "-u", f"{BOX}/relay.py", str(self.port)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  env=sandbox._sbx_env())    # stderr: sbx says "context canceled" when the relay is stopped
        first = self.p.stdout.readline()
        if '"ready"' not in first:
            raise RuntimeError("the model relay did not start inside the sandbox: " + first[:200])
        threading.Thread(target=self._loop, daemon=True).start()

    def api_base(self, key: str, iid: str) -> str:
        """The task's own endpoint: the relay knows which benchmark (harness) and task each call belongs to."""
        return f"http://127.0.0.1:{self.port}/t/{key}~{iid}/v1"

    def _loop(self):
        for line in self.p.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            threading.Thread(target=self._handle, args=(msg,), daemon=True).start()

    def _reply(self, rid, status: int, body: bytes, ctype="application/json"):
        with self.lock:
            try:
                self.p.stdin.write(json.dumps({"id": rid, "status": status, "ctype": ctype, "body": base64.b64encode(body).decode()}) + "\n")
                self.p.stdin.flush()
            except (OSError, ValueError):
                pass

    def _redact(self, text: str) -> str:
        """Error messages are saved (requests.jsonl, the log, the report): never with the API key, even if echoed back."""
        key = effective_key(self.s.base_url, self.s.api_key)
        if key:
            text = text.replace(key, "***")
        return _KEYLIKE.sub("***", text)

    def _record(self, row: dict):
        with self.slock, open(self.stats, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _handle(self, msg: dict):
        method, path = msg.get("method"), str(msg.get("path") or "").split("?")[0]
        if method == "GET" and re.match(r"^(/t/[^/]+)?/v1/models/?$", path):
            return self._reply(msg["id"], 200, json.dumps({"object": "list", "data": [{"id": "slmbench", "object": "model"}]}).encode())
        m = _PATH_RE.match(path)
        if method != "POST" or not m:
            return self._reply(msg["id"], 403, b'{"error":"blocked by slmbench"}')
        task = m.group(1)
        try:
            body = json.loads(base64.b64decode(msg.get("body") or "") or b"{}")
        except ValueError:
            return self._reply(msg["id"], 400, b'{"error":"invalid JSON"}')
        self._observations(task, body.get("messages") or [])
        body["model"] = self.model
        for k, v in self.extra.items():
            body.setdefault(k, v)
        body.setdefault("temperature", float(self.s.temperature))     # harnesses that don't send them get slmbench's
        if not body.get("max_tokens") and not body.get("max_completion_tokens"):
            body["max_tokens"] = limits(self.s)[1] if self.max_tokens is None else self.max_tokens
        headers = {"Content-Type": "application/json"}
        key = effective_key(self.s.base_url, self.s.api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        t0 = time.time()
        stream = bool(body.get("stream"))
        try:
            req = urllib.request.Request(self.s.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                         headers=headers, method="POST")
            with _OPENER.open(req, timeout=float(self.s.timeout or 600)) as r:
                data, ctype = r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            err = e.read()
            self.errors.append(self._redact(f"HTTP {e.code}: {err[:300].decode(errors='replace')}"))
            err = self._redact(err.decode(errors="replace")).encode()
            self._record({"error": f"HTTP {e.code}", "empty": True, "task": task.replace("~", "|")})
            return self._reply(msg["id"], e.code, err)
        except Exception as e:  # noqa: BLE001 - server down, timeout…
            text = self._redact(f"{type(e).__name__}: {e}")
            self.errors.append(text)
            self._record({"error": text[:200], "empty": True, "task": task.replace("~", "|")})
            return self._reply(msg["id"], 502, json.dumps({"error": text}).encode())
        self._reply(msg["id"], 200, data, ctype)
        self._account(task, data, stream, time.time() - t0)

    def _observations(self, task: str, messages: list) -> None:
        """What came back from the agent's previous step (command output, tool results, the harness's feedback) travels in
        the next request: the messages after the last assistant one. Saved in answers.jsonl, so the TUI can show the
        session while it happens (the harnesses write their own trajectory only at the end)."""
        last = max((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"), default=None)
        if last is None:
            return
        items = []
        for m in messages[last + 1:]:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
            items.append({"role": m.get("role"), "content": str(c or "")[:20000], "tool_call_id": m.get("tool_call_id") or "",
                          "name": m.get("name") or ""})
        if not items:
            return
        key, _, iid = task.partition("~")
        with self.slock:
            n = self.steps.get(task, 0)
            with open(self.stats.parent / "answers.jsonl", "a") as f:
                f.write(json.dumps({"key": key, "id": iid, "n": n, "kind": "obs", "items": items}, ensure_ascii=False) + "\n")

    def _account(self, task: str, data: bytes, stream: bool, elapsed: float):
        msgd, usage, timings, finish = {}, {}, {}, None
        try:
            if stream:                                      # SSE: rebuild the answer from the chunks
                text, think, calls = [], [], []
                for line in data.decode(errors="replace").splitlines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    ch = json.loads(line[5:])
                    usage = ch.get("usage") or usage
                    timings = ch.get("timings") or timings
                    for c in ch.get("choices") or []:
                        d = c.get("delta") or {}
                        text.append(d.get("content") or "")
                        think.append(d.get("reasoning_content") or d.get("reasoning") or "")
                        for tc in d.get("tool_calls") or []:
                            if (tc.get("function") or {}).get("name"):
                                calls.append({"function": {"name": tc["function"]["name"], "arguments": ""}})
                            if calls and (tc.get("function") or {}).get("arguments"):
                                calls[-1]["function"]["arguments"] += tc["function"]["arguments"]
                        finish = c.get("finish_reason") or finish
                msgd = {"content": "".join(text), "reasoning_content": "".join(think), "tool_calls": calls}
            else:
                d = json.loads(data)
                c = (d.get("choices") or [{}])[0]
                msgd, usage, timings, finish = c.get("message") or {}, d.get("usage") or {}, d.get("timings") or {}, c.get("finish_reason")
        except (ValueError, KeyError, TypeError):
            pass
        content = msgd.get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        calls = msgd.get("tool_calls") or []
        m = _BASH_RE.search(content)
        if calls:
            fn = calls[0].get("function") or {}
            args = fn.get("arguments") or ""
            try:
                a = json.loads(args) if isinstance(args, str) else args
                args = a.get("command") or a.get("path") or json.dumps(a, ensure_ascii=False) if isinstance(a, dict) else str(a)
            except ValueError:
                pass
            what = f"{fn.get('name', '?')}: {args}"
        elif m:
            what = "$ " + m.group(1).strip()
        else:
            what = T("(respuesta sin comando)", "(answer without a command)")
        what = " ".join(what.split())[:110]
        key, _, iid = task.partition("~")
        with self.slock:
            self.steps[task] = n = self.steps.get(task, 0) + 1
            u = self.usage.setdefault(task, {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0})
            u["turns"] += 1
            u["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            u["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            with open(self.stats.parent / "answers.jsonl", "a") as f:    # every answer, also the ones a harness rejects
                f.write(json.dumps({"key": key, "id": iid, "n": n, "finish": finish, "ct": usage.get("completion_tokens"),
                                    "content": content, "reasoning": msgd.get("reasoning_content") or msgd.get("reasoning") or "",
                                    "tool_calls": calls}, ensure_ascii=False) + "\n")
        self._record({"t": round(elapsed, 3), "ct": usage.get("completion_tokens"), "pt": usage.get("prompt_tokens"),
                      "tps": timings.get("predicted_per_second"), "pps": timings.get("prompt_per_second"), "finish": finish,
                      "empty": not (content.strip() or calls), "task": f"{key}|{iid}", "out": what})
        self.on_step(iid, key, n, what, finish)

    def stop(self):
        if self.p and self.p.poll() is None:
            try:
                self.p.stdin.close()
            except OSError:
                pass
            self.p.terminate()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()


# ---------------------------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------------------------
def server_context(base_url: str, api_key: str = "") -> int | None:
    """The context one request can use on the server (llama-server: n_ctx per slot, from /props); None if it doesn't say.
    With the API key: a server that wants one (Unsloth Studio) answers /props only with it."""
    from .servers import effective_key
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    key = effective_key(base_url, api_key)
    try:
        with _http(root + "/props", timeout=4, headers={"Authorization": f"Bearer {key}"} if key else None) as r:
            n = (json.loads(r.read()).get("default_generation_settings") or {}).get("n_ctx")
        return int(n) if n else None
    except (OSError, ValueError, TypeError):
        return None


def limits(s: Settings) -> tuple[int, int]:
    """(context, max tokens per answer) the harnesses are given. The context is the server's real one when it says so (else
    the setting); an answer may use at most a quarter of it, so the history always has room: a harness that reserves more
    for the answer than the window holds compacts or overflows at every step."""
    ctx = server_context(s.base_url, s.api_key) or int(s.max_context or 32768)
    return ctx, max(1024, min(int(s.agent_max_tokens), ctx // 4))


def _task_config(s: Settings, model: str, harness: str, inst: dict, image: str, api_base: str, label: str, run_id: str,
                 extra: dict | None = None, ctx: tuple[int, int] | None = None) -> dict:
    context, max_tokens = ctx or limits(s)
    return {"instance": inst, "harness": harness, "image": image, "api_base": api_base, "model": model, "label": run_id,
            "temperature": float(s.temperature), "max_tokens": max_tokens, "step_limit": int(s.agent_step_limit),
            "timeout": int(s.agent_timeout), "context": context, "memory": f"{TASK_MEMORY}g",
            "dir": f"{BOX}/tasks/{harness}/{inst['instance_id']}", "grader": f"{BOX}/grade.py",
            "pi_extension": f"{BOX}/pi_docker.ts", "opencode": "/opt/opencode/bin/opencode", **(extra or {})}


def _fetch_dir(box: str, src: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["sbx", "--no-input", "exec", box, "sh", "-c",
                        f"cd {src} && tar -cf - --exclude=./data --exclude=./mini-out --exclude=./pi-agent . && cd / && rm -rf {src}"],
                       capture_output=True, env=sandbox._sbx_env(), timeout=600)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("could not copy the task's results: " + p.stderr.decode(errors="replace")[-200:])
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tar:
        for m in tar.getmembers():                          # files and folders only, inside dest
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in Path(m.name).parts:
                continue
            tar.extract(m, dest, filter="data") if hasattr(tarfile, "data_filter") else tar.extract(m, dest)


def _steps_of(d: Path) -> int | None:
    from . import trajectory
    files = trajectory.trajectory_files(d)
    if not files:
        return None
    try:
        return len(trajectory.load(files[0])["steps"])
    except (OSError, ValueError):
        return None


def human(n) -> str:
    """412345 -> 412k, 1234567 -> 1.2M."""
    if n is None:
        return "-"
    n = float(n)
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}"


def _median(xs: list) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def usage_from_requests(run_dir: str | Path) -> dict:
    """{(benchmark key, task id): {turns, prompt_tokens, completion_tokens, tokens}} rebuilt from requests.jsonl, for runs
    whose tasks.jsonl doesn't carry them (every call there is tagged "key|task id")."""
    out: dict = {}
    try:
        lines = (Path(run_dir) / "requests.jsonl").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        key, _, iid = str(r.get("task") or "").partition("|")
        if not iid or "error" in r:
            continue
        u = out.setdefault((key, iid), {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "tokens": 0})
        u["turns"] += 1
        u["prompt_tokens"] += int(r.get("pt") or 0)
        u["completion_tokens"] += int(r.get("ct") or 0)
        u["tokens"] = u["prompt_tokens"] + u["completion_tokens"]
    return out


def task_stats(rows: list[dict], specs: list[str]) -> dict:
    """Per benchmark (harness): turns (model calls) and tokens per task, median and mean, over the tasks the agent worked on
    (an image that couldn't be fetched isn't a task the agent saw)."""
    out = {}
    for spec in specs:
        rs = [r for r in rows if r["spec"] == spec and not r.get("infra_error") and r.get("turns") is not None]
        if not rs:
            continue
        mean = lambda k: sum(r.get(k) or 0 for r in rs) / len(rs)      # noqa: E731
        out[spec] = {"tasks": len(rs), "turns_median": _median([r.get("turns") for r in rs]), "turns_mean": mean("turns"),
                     "tokens_median": _median([r.get("tokens") for r in rs]), "tokens_mean": mean("tokens"),
                     "prompt_tokens_mean": mean("prompt_tokens"), "completion_tokens_mean": mean("completion_tokens"),
                     "tokens_total": sum(r.get("tokens") or 0 for r in rs)}
    return out


def _summary(rows: list[dict], specs: list[str]) -> dict:
    out = {}
    for spec in specs:
        rs = [r for r in rows if r["spec"] == spec]
        if not rs:
            continue
        n = len(rs)

        def ratio(k):
            p = sum(bool(r.get(k)) for r in rs) / n
            return {"value": p, "stderr": math.sqrt(p * (1 - p) / n) if n > 1 else None}
        out[spec] = {"resolved": ratio("resolved"), "patch_applied": ratio("applied"), "submitted": ratio("patch")}
    return out


def run(s: Settings, model: str, specs: list[str], *, on_line=print, cancel: threading.Event | None = None,
        origin: str | None = None, reuse: Path | None = None, regrade: Path | None = None, server: dict | None = None) -> Path:
    """Runs the agent benchmarks; returns the run folder (same files as the other modes plus tasks.jsonl and agent/).
    reuse: an interrupted run whose finished tasks are kept (resume); regrade: a run whose patches are graded again without
    calling the model (rescore)."""
    from . import engine
    from .servers import clean_model_id
    why = status(refresh=True)
    if why:
        raise RuntimeError(T(f"El modo agente no está listo: {why}.", f"Agent mode isn't ready: {why}."))
    keys = [cat.spec_to_key(sp) for sp in specs]
    hs = harnesses()
    served = {}
    if not regrade:
        model = engine.resolve_model(s, model, on_line)
        try:
            from .servers import model_meta
            served = {k: v for k, v in model_meta(s.base_url, model, timeout=4, api_key=s.api_key).items() if v not in (None, "")}
        except Exception:  # noqa: BLE001
            served = {}
    custom_label = s.label if s.label and getattr(s, "label_model", "") in ("*", model) else ""
    label = custom_label or re.sub(r"[^A-Za-z0-9._-]+", "_", clean_model_id(model))
    kind = "rescore" if regrade else ("resume" if reuse else "")
    out = Path(s.out_dir).expanduser() / f"{label}{'-' + kind if kind else ''}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"model": model, "specs": specs, "kind": kind or "run", "origin": origin,
                                              "settings": {k: v for k, v in dataclasses.asdict(s).items() if k != "api_key"}},
                                             indent=1, ensure_ascii=False))
    stats, tasks_file = out / "requests.jsonl", out / "tasks.jsonl"
    log = open(out / "run.log", "a", buffering=1)
    _on_line = on_line

    def on_line(text):
        try:
            log.write(text + "\n")
        except (OSError, ValueError):
            pass
        _on_line(text)

    sandbox.prune()                                  # sandboxes left by a slmbench that died (kill -9, a crash)
    insts = select(s.ld_max_samples)
    avail_n, total_n = len(available()), cat.catalog()[keys[0]]["n"]
    work = [(inst, spec, hs[k]) for inst in insts for spec, k in zip(specs, keys)]
    done_rows: list[dict] = []
    prev = {}
    src = regrade or reuse
    if src:
        for line in (Path(src) / "tasks.jsonl").read_text().splitlines() if (Path(src) / "tasks.jsonl").exists() else []:
            try:
                r = json.loads(line)
                prev[(r["spec"], r["id"])] = r
            except (ValueError, KeyError):
                continue
    if regrade:
        work = [w for w in work if (w[1], w[0]["instance_id"]) in prev]
    elif reuse:                                                # resume: finished tasks are copied, not run again
        import shutil
        for inst, spec, h in list(work):
            r = prev.get((spec, inst["instance_id"]))
            sdir = Path(reuse) / "agent" / h / inst["instance_id"]
            if r and sdir.is_dir():
                shutil.copytree(sdir, out / "agent" / h / inst["instance_id"], dirs_exist_ok=True)
                done_rows.append({**r, "reused": True})
                with open(tasks_file, "a") as f:
                    f.write(json.dumps({**r, "reused": True}) + "\n")
                work.remove((inst, spec, h))
        if done_rows:
            on_line(T(f"Reanudando: {len(done_rows)} tareas ya terminadas se reutilizan", f"Resuming: {len(done_rows)} finished tasks reused"))
    n_all = len(work) + len(done_rows)
    run_id = f"{os.getpid()}-{secrets.token_hex(3)}"
    parallel = max(1, min(int(s.agent_workers or 1), MAX_PARALLEL))
    ctx_limits = limits(s)
    if ctx_limits[1] < int(s.agent_max_tokens):
        on_line(T(f"Contexto del servidor: {ctx_limits[0]} tokens → máx. {ctx_limits[1]} tokens por respuesta (no {s.agent_max_tokens}): "
                  "el resto queda para el historial", f"Server context: {ctx_limits[0]} tokens → max {ctx_limits[1]} tokens per answer "
                  f"(not {s.agent_max_tokens}): the rest is left for the history"))
    on_line(T(f"SWE-bench Lite · {len(insts)} tareas ({avail_n}/{total_n} con imagen para {arch()}) · "
              f"{', '.join(hs[k] for k in keys)} · {parallel} en paralelo",
              f"SWE-bench Lite · {len(insts)} tasks ({avail_n}/{total_n} with an image for {arch()}) · "
              f"{', '.join(hs[k] for k in keys)} · {parallel} in parallel"))
    active = engine._active_dir() / f"{os.getpid()}-{run_id}.json"
    cancel_file = out / ".cancel"                  # the live view of another slmbench asks to cancel by creating it
    cancel = cancel if cancel is not None else threading.Event()
    active.write_text(json.dumps({"pid": os.getpid(), "model": model, "label": label, "specs": specs, "stats": str(tasks_file),
                                  "out": str(out), "total": n_all, "started": time.time(),
                                  "per_task": {sp: len(insts) for sp in specs}, "base_url": s.base_url,
                                  "max_samples": s.ld_max_samples, "agent": True, "cancel_file": str(cancel_file)}))
    if not tasks_file.exists():
        tasks_file.touch()
    run_box = fetch_box = relay = None
    cancelled = False
    procs: set = set()
    plock = threading.Lock()
    kind_srv = engine.server_kind(s.base_url, served)
    progress = {"done": len(done_rows)}

    def on_step(iid, key, n, what, finish):
        cut = " ✂" if finish == "length" else ""
        on_line(f"  {iid} [{hs.get(key, key)}] · #{n}{cut} {what}")

    try:
        on_line(T("Creando los sandboxes (ejecución sin red + descarga de imágenes)…", "Creating the sandboxes (no-network run + image fetcher)…"))
        with ThreadPoolExecutor(2) as ex:
            fr = ex.submit(_create, "a", TEMPLATE, run_memory(parallel))
            ff = ex.submit(_create, "f", None, FETCH_MEMORY)
            run_box = fr.result()
            fetch_box = ff.result()
        on_line(T(f"Sandbox {run_box} (sin red, {run_memory(parallel)} de RAM) y {fetch_box} (imágenes, {FETCH_MEMORY}); se eliminan al terminar",
                  f"Sandbox {run_box} (no network, {run_memory(parallel)} RAM) and {fetch_box} (images, {FETCH_MEMORY}); both removed when the run ends"))
        _upload(run_box, {n: (HERE / n).read_bytes() for n in ("relay.py", "task.py", "grade.py", "pi_docker.ts")}, BOX)
        images = Images(run_box, fetch_box)
        if not regrade:
            relay = Relay(run_box, s, model, stats, on_step, kind_srv)
            relay.max_tokens = ctx_limits[1]
            relay.start()
        by_inst: dict[str, list] = {}
        for inst, spec, h in work:
            by_inst.setdefault(inst["instance_id"], []).append((inst, spec, h))
        queue = list(by_inst.values())
        qlock = threading.Lock()
        avail = available()

        def one(inst, spec, h):
            iid = inst["instance_id"]
            image = avail[iid]
            extra = {}
            if regrade:
                pr = prev[(spec, iid)]
                pdir = Path(regrade) / "agent" / h / iid
                extra = {"harness": "regrade", "patch": (pdir / "patch.diff").read_text() if (pdir / "patch.diff").exists() else "",
                         "exit_status": pr.get("exit_status")}
            cfg = _task_config(s, model, h, inst, image, relay.api_base(cat.spec_to_key(spec), iid) if relay else "", label, run_id, extra,
                               ctx_limits)
            if regrade:                                     # the earlier run's files, graded again in this run's folder
                import shutil
                shutil.copytree(Path(regrade) / "agent" / h / iid, out / "agent" / h / iid, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("report.json", "test_output.txt"))
            p = subprocess.Popen(["sbx", "--no-input", "exec", "-i", run_box, "python3", "-u", f"{BOX}/task.py"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                 env=sandbox._sbx_env(), start_new_session=True)
            with plock:
                procs.add(p)
            p.stdin.write(json.dumps(cfg) + "\n")
            p.stdin.close()
            result = None
            for line in p.stdout:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("ev") == "phase" and ev.get("phase") == "grading":
                    on_line(T(f"  {iid} [{h}] · evaluando el parche con SWE-bench…", f"  {iid} [{h}] · grading the patch with SWE-bench…"))
                elif ev.get("ev") == "done":
                    result = ev
            p.wait()
            with plock:
                procs.discard(p)
            if cancel is not None and cancel.is_set():
                return
            tdir = out / "agent" / h / iid
            try:
                _fetch_dir(run_box, cfg["dir"], tdir)
            except RuntimeError as e:
                on_line(f"  ⚠ {iid}: {e}")
            if result is None:
                err = (p.stderr.read() or "").strip()[-300:]
                result = {"exit_status": f"Error: {err or 'task runner failed'}", "patch": False, "resolved": False, "report": {}}
            rep = result.get("report") or {}
            ts = rep.get("tests_status") or {}
            f2p = ts.get("FAIL_TO_PASS") or {}
            if regrade:
                use = {k: prev[(spec, iid)].get(k) for k in ("turns", "prompt_tokens", "completion_tokens", "tokens")}
            else:
                u = dict(relay.usage.get(f"{cat.spec_to_key(spec)}~{iid}") or {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0})
                use = {**u, "tokens": u["prompt_tokens"] + u["completion_tokens"]}
            row = {"spec": spec, "harness": h, "id": iid, "repo": inst.get("repo"), "resolved": bool(result.get("resolved")),
                   "applied": bool(rep.get("patch_successfully_applied")), "patch": bool(result.get("patch")),
                   "exit_status": result.get("exit_status"), "steps": _steps_of(tdir) if not regrade else prev[(spec, iid)].get("steps"),
                   "f2p": [len(f2p.get("success") or []), len(f2p.get("success") or []) + len(f2p.get("failure") or [])],
                   "p2p_broken": len((ts.get("PASS_TO_PASS") or {}).get("failure") or []),
                   "agent_time": result.get("agent_time") if not regrade else prev[(spec, iid)].get("agent_time"),
                   "total_time": result.get("total_time") if not regrade else prev[(spec, iid)].get("total_time"),
                   "grading_error": rep.get("error"), **use}
            with plock:
                done_rows.append(row)
                progress["done"] += 1
                with open(tasks_file, "a") as f:
                    f.write(json.dumps(row) + "\n")
                k = progress["done"]
            res = sum(r["resolved"] for r in done_rows if r["spec"] == spec)
            nn = sum(1 for r in done_rows if r["spec"] == spec)
            on_line(("✔ " if row["resolved"] else "✘ ") + f"{iid} [{h}] · "
                    + (T("resuelta", "resolved") if row["resolved"] else T("no resuelta", "unresolved"))
                    + f" · {row['steps'] or '?'} " + T("pasos", "steps") + f" · {row['exit_status']}"
                    + (f" · F2P {row['f2p'][0]}/{row['f2p'][1]}" if row["f2p"][1] else "")
                    + (f" · P2P ✘{row['p2p_broken']}" if row["p2p_broken"] else "")
                    + (f" · {row.get('turns')} " + T("turnos", "turns") + f" · {human(row.get('tokens'))} tokens" if row.get("turns") else "")
                    + f" · {int(row['total_time'] or 0)} s   [{k}/{n_all} · {res}/{nn} " + T("resueltas", "resolved") + "]")

        def worker():
            while True:
                if cancel is not None and cancel.is_set():
                    return
                with qlock:
                    if not queue:
                        return
                    group = queue.pop(0)
                inst = group[0][0]
                image = avail[inst["instance_id"]]
                on_line(T(f"→ {inst['instance_id']}: descargando la imagen…", f"→ {inst['instance_id']}: fetching the image…"))
                try:
                    images.fetch(image)
                except (RuntimeError, subprocess.SubprocessError) as e:
                    on_line(f"  ⚠ {inst['instance_id']}: {e}")
                    for _inst, spec, h in group:
                        row = {"spec": spec, "harness": h, "id": inst["instance_id"], "repo": inst.get("repo"), "resolved": False,
                               "applied": False, "patch": False, "exit_status": f"ImageError: {str(e)[:200]}", "steps": 0,
                               "f2p": [0, 0], "p2p_broken": 0, "agent_time": 0, "total_time": 0, "infra_error": True,
                               "turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "tokens": 0}
                        with plock:
                            done_rows.append(row)
                            progress["done"] += 1
                            with open(tasks_file, "a") as f:
                                f.write(json.dumps(row) + "\n")
                    continue
                try:
                    for inst_, spec, h in group:
                        if cancel is not None and cancel.is_set():
                            return
                        on_line(T(f"→ {inst_['instance_id']} [{h}]: el agente trabaja…", f"→ {inst_['instance_id']} [{h}]: the agent is working…"))
                        one(inst_, spec, h)
                finally:
                    images.remove(image)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(parallel)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            try:
                for t in threads:
                    t.join(timeout=0.5)
                if cancel_file.exists():
                    cancel.set()
                if cancel is not None and cancel.is_set() and not cancelled:
                    cancelled = True
                    on_line(T("Cancelando…", "Cancelling…"))
                    with plock:
                        for p in list(procs):
                            try:
                                os.killpg(p.pid, 15)
                            except OSError:
                                pass
            except KeyboardInterrupt:
                if cancel is None:
                    cancel = threading.Event()
                cancel.set()
        cancelled = cancelled or bool(cancel is not None and cancel.is_set())
    finally:
        if relay:
            relay.stop()
        for b in (run_box, fetch_box):
            if b:
                _forget(b)
        if run_box:
            on_line(T("Sandboxes eliminados", "Sandboxes removed"))
        for f in (active, cancel_file):
            try:
                f.unlink()
            except OSError:
                pass
        log.close()
    summary = _summary(done_rows, specs)
    agent_stats = task_stats(done_rows, specs)
    meta = {"model": model, "base_url": s.base_url, "specs": specs, "date": dt.datetime.now().isoformat(timespec="seconds"),
            "label": label, "mode": "agent", "temperature": s.temperature, "reasoning_budget": s.reasoning_budget,
            "max_new_tokens": ctx_limits[1], "context": ctx_limits[0], "step_limit": s.agent_step_limit, "task_timeout": s.agent_timeout,
            "parallel": parallel, "tasks": len(insts), "available": avail_n, "dataset_n": total_n, "arch": arch(),
            "images": "Epoch AI arm64 (ghcr.io/epoch-research)" if arch() == "arm64" else "SWE-bench (docker.io/swebench)",
            "dataset": dataset_info(), "harness_versions": {"mini-swe-agent": MINI_SWE_AGENT, "pi": pi_version(), "opencode": OPENCODE, "swebench": SWEBENCH},
            # partial unless every task of the dataset ran (on arm64 the 48 tasks without an image make every run partial)
            "max_samples": 0 if len(insts) >= total_n else len(insts),
            "returncode": 0, "cancelled": cancelled, "api_errors": len(relay.errors) if relay else 0,
            "api_error_examples": (relay.errors[:3] if relay else []), "api_notes": [],
            "server_kind": kind_srv, "kind": kind or "run", "origin": origin, "server": served or server or {},
            "custom_label": custom_label, "agent_stats": agent_stats, **engine.request_stats(stats)}
    meta["leaderboard"] = None
    if summary and not cancelled and s.leaderboard_auto:
        try:
            from . import leaderboard
            leaderboard.record(s, meta, summary)
            meta["leaderboard"] = str(leaderboard.paths(s)["md"])
        except Exception as e:  # noqa: BLE001 - the leaderboard never breaks a run
            meta["leaderboard_error"] = f"{type(e).__name__}: {e}"
    (out / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report(meta, summary, done_rows))
    return out


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def report(meta: dict, summary: dict, rows: list[dict]) -> str:
    from .engine import pct
    from .trajectory import pretty_status
    c = cat.catalog()
    budget = T("sin límite", "unlimited") if meta.get("reasoning_budget", -1) < 0 else f"≤ {meta['reasoning_budget']} tokens"
    hv = meta.get("harness_versions") or {}
    ds = meta.get("dataset") or {}
    lora = (meta.get("server") or {}).get("lora")
    L = [f"# {T('Informe', 'Report')}: {meta['label']}", "",
         f"- {T('Modelo', 'Model')}: `{meta['model']}`" + (f" + LoRA `{lora}`" if lora else "") + f"  ·  {T('Servidor', 'Server')}: `{meta['base_url']}`",
         f"- {T('Fecha', 'Date')}: {meta['date']}  ·  harness: "
         + ", ".join(f"{c[cat.spec_to_key(sp)]['harness']} {hv.get(c[cat.spec_to_key(sp)]['harness'], '')}" for sp in meta["specs"])
         + f"  ·  {T('evaluación', 'grading')}: swebench {hv.get('swebench')}",
         f"- SWE-bench Lite ({ds.get('dataset', DATASET)} @ {str(ds.get('revision', '?'))[:10]}): {meta['tasks']} "
         + T("tareas", "tasks") + f" · {meta['available']}/{meta['dataset_n']} " + T(f"con imagen para {meta['arch']}", f"with an image for {meta['arch']}")
         + f" ({meta.get('images')})",
         f"- {T('temperatura', 'temperature')} {meta['temperature']}  ·  " + (f"{T('contexto', 'context')} {meta['context']}  ·  "
                                                                              if meta.get("context") else "")
         + f"{T('máx. tokens por respuesta', 'max tokens per answer')} {meta['max_new_tokens']}  ·  {T('razonamiento', 'reasoning')}: {budget}  ·  {T('máx. pasos', 'max steps')} {meta['step_limit']}  ·  "
         f"{T('tiempo por tarea', 'time per task')} {int(meta['task_timeout']) // 60} min  ·  {meta['parallel']} " + T("en paralelo", "in parallel")]
    if meta.get("requests"):
        L.append(f"- {T('Llamadas al modelo', 'Model calls')}: {meta['requests']}  ·  {T('cortadas por max tokens', 'cut by max tokens')}: "
                 f"{meta.get('truncated', 0)}  ·  {T('generación', 'generation')}: {meta.get('gen_tps') or '-'} tok/s  ·  prefill: "
                 f"{meta.get('prompt_tps') or '-'} tok/s")
    if meta.get("max_samples"):
        L.append("- ⚠ " + T(f"Ejecución parcial ({meta['tasks']} de {meta['dataset_n']} tareas): cifras orientativas, no comparables con la literatura.",
                            f"Partial run ({meta['tasks']} of {meta['dataset_n']} tasks): indicative numbers, not comparable with published results."))
    if meta.get("kind") == "rescore":
        L.append("- " + T(f"Reevaluada a partir de `{meta.get('origin')}` con sus parches guardados (sin llamar al modelo).",
                          f"Regraded from `{meta.get('origin')}` using its saved patches (no calls to the model)."))
    elif meta.get("kind") == "resume":
        L.append("- " + T(f"Reanudada a partir de `{meta.get('origin')}`.", f"Resumed from `{meta.get('origin')}`."))
    if meta.get("cancelled"):
        L.append("- ⚠ " + T("Cancelada por el usuario.", "Cancelled by the user."))
    if meta.get("api_errors"):
        L.append("- ⚠ " + T(f"**Llamadas al modelo fallidas: {meta['api_errors']}**. Ejemplo: `{(meta.get('api_error_examples') or [''])[0][:120]}`",
                            f"**Failed model calls: {meta['api_errors']}**. Example: `{(meta.get('api_error_examples') or [''])[0][:120]}`"))
    infra = [r for r in rows if r.get("infra_error") or r.get("grading_error")]
    if infra:
        L.append("- ⚠ " + T(f"{len(infra)} tareas con error de infraestructura (imagen o evaluación): cuentan como no resueltas.",
                            f"{len(infra)} tasks with an infrastructure error (image or grading): they count as unresolved."))
    L += ["", f"## {T('Resumen', 'Summary')}", "",
          f"| Benchmark | {T('Resueltas', 'Resolved')} | ± | {T('Parche aplicado', 'Patch applied')} | "
          f"{T('Con parche', 'With a patch')} | {T('Turnos/tarea', 'Turns/task')} | {T('Tokens/tarea', 'Tokens/task')} |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for spec, m in summary.items():
        k = cat.spec_to_key(spec)
        rs = [r for r in rows if r["spec"] == spec]
        st = (meta.get("agent_stats") or {}).get(spec) or {}
        turns = f"{st['turns_median']:g}" if st.get("turns_median") is not None else "-"
        toks = human(st.get("tokens_median")) if st.get("tokens_median") is not None else "-"
        L.append(f"| {c[k]['name']} | {sum(r['resolved'] for r in rs)}/{len(rs)} ({pct(m['resolved']['value'])}) | "
                 f"{pct(m['resolved']['stderr']) if m['resolved'].get('stderr') is not None else '-'} | {pct(m['patch_applied']['value'])} | "
                 f"{pct(m['submitted']['value'])} | {turns} | {toks} |")
    for spec in summary:
        k = cat.spec_to_key(spec)
        rs = sorted([r for r in rows if r["spec"] == spec], key=lambda r: (not r["resolved"], r["id"]))
        L += ["", f"## {c[k]['name']} · {T('tareas', 'tasks')}", "",
              f"| {T('Tarea', 'Task')} | {T('Resuelta', 'Resolved')} | {T('Final', 'Exit')} | {T('Turnos', 'Turns')} | Tokens | "
              f"FAIL_TO_PASS | {T('P2P rotos', 'P2P broken')} | {T('Tiempo', 'Time')} |", "|---|:-:|---|---:|---:|---:|---:|---:|"]
        for r in rs:
            f2p = f"{r['f2p'][0]}/{r['f2p'][1]}" if r.get("f2p") and r["f2p"][1] else "-"
            turns = r.get("turns") if r.get("turns") is not None else "-"
            toks = (f"{human(r.get('tokens'))} <sub>({human(r.get('completion_tokens'))} gen.)</sub>"
                    if r.get("tokens") is not None else "-")
            L.append(f"| `{r['id']}` | {'✔' if r['resolved'] else '✘'} | {pretty_status(r.get('exit_status'))[:40]} | {turns} | {toks} | "
                     f"{f2p} | {r.get('p2p_broken') or 0} | {int(r.get('total_time') or 0)} s |")
    L += ["", "> " + T("Turnos = llamadas al modelo del harness en la tarea (también las respuestas rechazadas y las compactaciones); "
                       "tokens = de entrada + generados en todas ellas (la entrada crece cada turno: se reenvía el historial).",
                       "Turns = the harness's model calls for the task (rejected answers and compactions included); tokens = prompt + "
                       "generated over all of them (the prompt grows every turn: the history is sent again).")]
    L += ["", "> " + T("Cada paso del agente (razonamiento, respuesta, comando y salida): tecla `e` en Resultados o `slmbench traj <ejecución>`.",
                       "Every step of the agent (reasoning, answer, command and output): key `e` in Results or `slmbench traj <run>`."), ""]
    return "\n".join(L)
