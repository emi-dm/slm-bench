"""Where model-generated code runs: always inside a Docker Sandbox (`sbx`), never on the host.

Every benchmark that executes code (HumanEval+, MBPP+, CRUXEval, MultiPL-E, LiveCodeBench) goes through `execute(files,
steps)`: the files are written to a fresh temporary directory *inside the sandbox*, the steps (e.g. compile, then run) run
in order with their own timeouts, and the directory is deleted afterwards. There is no host fallback: if the sandbox is
not available, coding benchmarks don't run (slmbench refuses to start them), and a check that can't reach the sandbox
scores 0.

The sandbox, `slmbench-exec`, is created by `slmbench setup --mode code`: a microVM with its own kernel, no host
workspace (it can't see your files), a CPU/memory cap and a deny-all network rule. The Debian mirror is allowed only
while the compilers are installed and the rule is removed right after. Each `execute` is one `sbx exec` call; the files
travel as a tar on stdin.

Inside the sandbox, model code never runs as the sandbox's default user (`agent`, which has sudo and the docker group).
Each check runs as one of the unprivileged users `slmrun0…7` (no sudo, no groups, no home, `no_new_privs` so setuid
binaries can't raise it), in its own directory that only that user can open, with limits on file size and number of
processes. A slot (= user) is taken with a lock, so two checks never share a user; when the check ends every process of
that user is killed (also anything left in the background) and its temporary files are removed. The model's code can't
change the tools, the scoring harness or other checks.
"""
from __future__ import annotations

import atexit
import base64
import threading
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

NAME = os.environ.get("SLMBENCH_SANDBOX_NAME", "slmbench-exec")
PACKAGES = ["nodejs", "g++", "default-jdk-headless", "golang-go", "rustc", "python3-numpy", "python3-tqdm"]
MIRROR = "deb.debian.org"
MARKER = "/etc/slmbench-sandbox"
VERSION = "2"                        # bump to re-provision existing sandboxes when PACKAGES or the hardening change
SLOTS = 8                            # unprivileged users slmrun0..7: at most this many checks at the same time
JOBS = "/home/agent/slmjobs"         # per-check directories (0711: a user can enter only its own)
LOCKS = "/home/agent/.slmslots"      # slot locks (only agent can create them)
FSIZE = 128 * 1024 * 1024            # largest file a check may write
NPROC = 512                          # ulimit -u per slot user: no fork bombs
# run as root once per sandbox (no network needed): the users and directories the hardening relies on
HARDEN = ("set -e; for i in $(seq 0 %d); do id -u slmrun$i >/dev/null 2>&1 || useradd --system --no-create-home "
          "--home-dir /nonexistent --shell /usr/sbin/nologin --user-group slmrun$i; done; "
          "install -d -o agent -g agent -m 0711 %s; install -d -o agent -g agent -m 0700 %s") % (SLOTS - 1, JOBS, LOCKS)
TOOLS = ["python3", "node", "g++", "javac", "java", "go", "rustc"]
# logical tool name -> name inside the sandbox
SBX_TOOLS = {"python": "python3", "node": "node", "c++": "g++", "javac": "javac", "java": "java", "go": "go", "rustc": "rustc"}
ENV = {"LANG": "C.UTF-8", "GOTOOLCHAIN": "local", "GOPROXY": "off", "GOFLAGS": "-mod=mod", "GO111MODULE": "on"}


_status: dict = {}


def status(refresh: bool = False) -> str | None:
    """None if model code can run (the sandbox is ready); otherwise a short reason. Cached per process."""
    if refresh or "s" not in _status:
        if not sbx_installed():
            _status["s"] = "Docker Sandboxes (sbx) not installed"
        elif not daemon_ok():
            _status["s"] = "sbx daemon not running"
        elif not ready() and not _upgrade():
            _status["s"] = "sandbox not set up (slmbench setup --mode code)"
        else:
            _status["s"] = None
    return _status["s"]


# ---------------------------------------------------------------------------------------------
# sbx management
# ---------------------------------------------------------------------------------------------
_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.I)


def _sbx_env() -> dict:
    """The environment `sbx` is started with: the user's, minus anything that looks like a secret (the API key of the model
    under test lives in SLMBENCH_LD_API_KEY while lighteval runs). The sandbox never needs them."""
    return {k: v for k, v in os.environ.items() if not _SECRET.search(k)}


def _sbx(*args, timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["sbx", "--no-input", *args], capture_output=True, text=True, timeout=timeout, env=_sbx_env(), **kw)


def sbx_installed() -> bool:
    return shutil.which("sbx") is not None


def daemon_ok() -> bool:
    try:
        return sbx_installed() and _sbx("ls", "--json", timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def exists(name: str = NAME) -> bool:
    try:
        r = _sbx("ls", "--json", timeout=30)
        return r.returncode == 0 and any(s.get("name") == name for s in json.loads(r.stdout or "[]"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def ready(name: str = NAME) -> bool:
    """The sandbox exists and was provisioned with this version of the package list."""
    try:
        r = _sbx("exec", name, "cat", MARKER, timeout=120)
        return r.returncode == 0 and r.stdout.strip() == VERSION
    except (OSError, subprocess.SubprocessError):
        return False


def _upgrade(name: str = NAME) -> bool:
    """A sandbox provisioned by an older slmbench (version 1: compilers, but model code ran as `agent`) is hardened in
    place: creates the unprivileged users and directories. No network, no reinstall."""
    try:
        r = _sbx("exec", name, "cat", MARKER, timeout=120)
        if r.returncode != 0 or r.stdout.strip() != "1":
            return False
        h = _sbx("exec", "-u", "root", name, "sh", "-c", HARDEN + f" && echo {VERSION} > {MARKER}", timeout=300)
        return h.returncode == 0 and ready(name)
    except (OSError, subprocess.SubprocessError):
        return False


def missing_tools(tools: list[str], name: str = NAME) -> list[str]:
    """Logical tool names (node, c++, javac, java, go, rustc) not available inside the sandbox."""
    wanted = {t: SBX_TOOLS.get(t, t) for t in tools}
    script = "for t in " + " ".join(shlex.quote(v) for v in wanted.values()) + "; do command -v \"$t\" >/dev/null || echo \"$t\"; done"
    try:
        r = _sbx("exec", name, "sh", "-c", script, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return list(tools)
    if r.returncode != 0:
        return list(tools)
    absent = set(r.stdout.split())
    return [t for t, v in wanted.items() if v in absent]


def default_cpus() -> int:
    """Half of the host's cores (at least 2, at most 8): tests run fast and the host stays responsive."""
    return max(2, min(8, (os.cpu_count() or 4) // 2))


def provision(on_line=print, name: str = NAME, cpus: int | None = None, memory: str = "4g") -> bool:
    """Creates and prepares the sandbox: compilers via apt (only the Debian mirror allowed, only for this sandbox, only
    meanwhile), then a deny-all network rule. Idempotent."""
    if not sbx_installed():
        on_line("✗ Docker Sandboxes (`sbx`) is not installed: https://docs.docker.com/ai/sandboxes/get-started/")
        return False
    if not daemon_ok():
        on_line("✗ The Docker Sandboxes daemon is not answering: run `sbx daemon start` (or `sbx diagnose`)")
        return False
    if ready(name) or _upgrade(name):
        on_line(f"✔ sandbox {name} ready (no network, model code runs as unprivileged users)")
        return True
    cpus = cpus or default_cpus()
    if not exists(name):
        on_line(f"  creating sandbox {name} ({cpus} CPUs, {memory}, no host workspace)…")
        r = _sbx("create", "--no-progress", "--name", name, "--cpus", str(cpus), "--memory", memory, "shell", timeout=900)
        if r.returncode != 0:
            on_line("✗ " + (r.stderr or r.stdout).strip()[-400:])
            return False
    on_line(f"  installing compilers inside the sandbox (temporarily allowing only {MIRROR})…")
    r = _sbx("policy", "allow", "network", "--sandbox", name, MIRROR, timeout=60)
    rule = re.search(r"Added rule (\S+)", r.stdout + r.stderr)
    try:
        script = ("export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/tmp/apt.log 2>&1 && "
                  f"apt-get install -y -qq --no-install-recommends {' '.join(PACKAGES)} >>/tmp/apt.log 2>&1; rc=$?; "
                  "grep -E '^E:' /tmp/apt.log | head -3; exit $rc")
        r = _sbx("exec", "-u", "root", name, "sh", "-c", script, timeout=1800)
        ok = r.returncode == 0
        if not ok:
            on_line("✗ apt failed: " + (r.stdout + r.stderr).strip()[-400:])
    finally:
        if rule:
            _sbx("policy", "rm", "network", "--force", "--sandbox", name, "--id", rule.group(1), timeout=60)
        deny = _sbx("policy", "deny", "network", "--sandbox", name, "**", timeout=60)
        if deny.returncode != 0 and "already" not in (deny.stdout + deny.stderr).lower():
            on_line("⚠ could not add the deny-all network rule: " + (deny.stdout + deny.stderr).strip()[-300:])
    if not ok:
        return False
    h = _sbx("exec", "-u", "root", name, "sh", "-c", HARDEN + f" && echo {VERSION} > {MARKER}", timeout=300)
    if h.returncode != 0:
        on_line("✗ could not create the unprivileged users: " + (h.stdout + h.stderr).strip()[-300:])
        return False
    still = missing_tools(["node", "c++", "javac", "java", "go", "rustc"], name)
    if still:
        on_line("✗ still missing inside the sandbox: " + ", ".join(still))
        return False
    on_line(f"✔ sandbox {name} ready: python3, node, g++, javac/java, go, rustc · network: denied · "
            "model code runs as unprivileged users")
    return True


def network_blocked(name: str = NAME) -> bool:
    """Checks from inside the sandbox that the outside world is unreachable."""
    code = ("import urllib.request\nfor u in ('https://pypi.org/simple/', 'https://github.com/'):\n"
            "    try:\n        urllib.request.urlopen(u, timeout=8); print('REACHABLE')\n    except Exception:\n        pass\n")
    try:
        r = _sbx("exec", name, "python3", "-c", code, timeout=120)
        return r.returncode == 0 and "REACHABLE" not in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


# ---------------------------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------------------------
def execute(files: dict[str, str | bytes], steps: list[tuple[list[str], float, str]]) -> tuple[bool, str]:
    """Writes `files` into a fresh temporary directory and runs `steps` = [(argv, timeout_s, label), ...] in order,
    stopping at the first failure, all inside the sandbox. argv[0] is a logical tool (python, node, c++, javac, java, go,
    rustc) or a path relative to the directory (./prog). Returns (all steps succeeded, output); a failed compile step is
    reported as "compile: …", a timeout as "timeout"."""
    ok_out = _worker().run(files, steps)
    if ok_out is not None:
        return ok_out
    return _execute_sbx(files, steps)                  # fallback: one `sbx exec` per check (still in the sandbox)


_warned: set[str] = set()


def _warn_once(msg: str):
    if msg not in _warned:
        _warned.add(msg)
        print(f"SLMBENCH_SANDBOX_WARNING {msg}", file=sys.stderr, flush=True)


def _tar(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path, data in files.items():
            raw = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(raw), 0o644
            tf.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def _as_runner_prefix() -> str:
    """Shell: run "$@" as the slot user $u with no privileges, a clean environment and resource limits."""
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in ENV.items())
    return ('sudo -n /usr/bin/setpriv --reuid="$u" --regid="$u" --clear-groups --no-new-privs '
            f'/usr/bin/env -i PATH="$PATH" HOME="$d" TMPDIR="$d/tmp" {envs} '
            f'/usr/bin/prlimit --fsize={FSIZE} --nproc={NPROC} /bin/sh -c \'cd "$HOME" && exec "$@"\' sh')


def _execute_sbx(files, steps) -> tuple[bool, str]:
    lines = [
        # take a free slot (= an unprivileged user) with a lock; wait for slot 0 if all are busy
        f'slot=; for i in $(seq 0 {SLOTS - 1}); do exec 9>{LOCKS}/$i; if flock -n 9; then slot=$i; break; fi; done',
        f'[ -n "$slot" ] || {{ exec 9>{LOCKS}/0; flock 9; slot=0; }}',
        'u=slmrun$slot',
        f'd=$(mktemp -d {JOBS}/slm.XXXXXX) || exit 97', 'cd "$d" || exit 97',
        "tar -xf - || { echo __SLM_FAIL__ untar 98; exit 98; }",
        'mkdir -p tmp && sudo -n chown -R "$u:$u" "$d" || { echo __SLM_FAIL__ setup 97; exit 97; }', 'cd /',
        'cleanup() { cd /; n=0; while [ $n -lt 100 ]; do sudo -n pkill -STOP -u "$u" 2>/dev/null; '
        'sudo -n pkill -KILL -u "$u" 2>/dev/null || break; n=$((n+1)); done; sudo -n rm -rf "$d"; '
        'sudo -n find /tmp /var/tmp /dev/shm -xdev -user "$u" -delete 2>/dev/null; }',
        # model code gets a clean environment: no sandbox credential placeholders or other variables
        'step() { label=$1; t=$2; shift 2; timeout -k 2 "$t" ' + _as_runner_prefix() + ' "$@"; rc=$?; '
        'if [ $rc -ne 0 ]; then echo "__SLM_FAIL__ $label $rc"; cleanup; exit $rc; fi; }']
    total = 10.0
    for argv, t, label in steps:
        tool = SBX_TOOLS.get(argv[0], argv[0])
        lines.append("step " + " ".join(shlex.quote(x) for x in [label, str(int(t) + 1), tool, *argv[1:]]))
        total += t + 3
    lines.append("cleanup")
    try:
        p = subprocess.run(["sbx", "exec", "-i", NAME, "sh", "-c", "\n".join(lines)], input=_tar(files), env=_sbx_env(),
                           capture_output=True, timeout=total + 60)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as e:
        _warn_once(f"cannot run sbx: {e}")
        return False, f"sandbox unavailable: {e}"
    out = (p.stdout + p.stderr).decode(errors="replace")
    if p.returncode == 125 or (p.returncode != 0 and "__SLM_FAIL__" not in out and "sandbox" in out.lower() and "not" in out.lower()):
        _warn_once(f"sandbox {NAME} unavailable ({out.strip()[-200:]}): run `slmbench setup --mode code`")
        return False, "sandbox unavailable"
    m = re.search(r"__SLM_FAIL__ (\S+) (\d+)", out)
    body = re.sub(r"__SLM_FAIL__ \S+ \d+\n?", "", out).strip()
    if p.returncode == 0 and not m:
        return True, body[-2000:]
    if m and m.group(2) in ("124", "137"):
        return False, "timeout"
    if m and m.group(1) == "compile":
        return False, "compile: " + body[-2000:]
    return False, body[-2000:]


# ---------------------------------------------------------------------------------------------
# persistent worker (speed): one long-lived `sbx exec` that runs many checks, several at a time
# ---------------------------------------------------------------------------------------------
# Runs *inside* the sandbox. Reads one JSON request per line on stdin ({id, tar (base64), steps}), runs it in a fresh temp
# dir with a clean environment and a timeout per step (killing the whole process group), and writes {id, ok, out} lines.
WORKER = r"""
import base64, fcntl, io, json, os, shutil, subprocess, sys, tarfile, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
TOOLS = %(tools)s
ENV = %(env)s
SLOTS, JOBS, LOCKS, FSIZE, NPROC = %(slots)d, %(jobs)r, %(locks)r, %(fsize)d, %(nproc)d
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
lock = threading.Lock()
def emit(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
def take_slot():
    # a slot = an unprivileged user; flock makes it exclusive across threads *and* other slmbench processes
    while True:
        for i in range(SLOTS):
            fd = os.open(f"{LOCKS}/{i}", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return i, fd
            except OSError:
                os.close(fd)
        time.sleep(0.05)
def sudo(*argv):
    return subprocess.run(["sudo", "-n", *argv], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def kill_all(user):
    # freeze, then kill, until nothing of this user is left (a fork bomb keeps spawning between two passes)
    for _ in range(100):
        sudo("pkill", "-STOP", "-u", user)
        if sudo("pkill", "-KILL", "-u", user).returncode != 0:   # 1 = no process matched
            return
        time.sleep(0.02)
def cleanup(user, d):
    kill_all(user)                                           # everything the check left running, background included
    sudo("rm", "-rf", d)
    sudo("find", "/tmp", "/var/tmp", "/dev/shm", "-xdev", "-user", user, "-delete")
def handle(req):
    slot, fd = take_slot()
    user = f"slmrun{slot}"
    d = tempfile.mkdtemp(prefix="slm.", dir=JOBS)
    try:
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(req["tar"]))) as tf:
            tf.extractall(d, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(d)
        os.makedirs(f"{d}/tmp", exist_ok=True)
        if sudo("chown", "-R", f"{user}:{user}", d).returncode != 0:
            return emit({"id": req["id"], "ok": False, "out": "worker error: cannot hand the directory to " + user})
        envs = [f"{k}={v}" for k, v in {"PATH": PATH, "HOME": d, "TMPDIR": f"{d}/tmp", **ENV}.items()]
        prefix = ["sudo", "-n", "/usr/bin/setpriv", f"--reuid={user}", f"--regid={user}", "--clear-groups", "--no-new-privs",
                  "/usr/bin/env", "-i", *envs, "/usr/bin/prlimit", f"--fsize={FSIZE}", f"--nproc={NPROC}",
                  "/bin/sh", "-c", 'cd "$HOME" && exec "$@"', "sh"]
        out = ""
        for argv, t, label in req["steps"]:
            argv = [TOOLS.get(argv[0], argv[0])] + argv[1:]
            try:
                p = subprocess.Popen(prefix + argv, cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                return emit({"id": req["id"], "ok": False, "out": f"cannot run {argv[0]}: {e}"})
            try:
                raw, _ = p.communicate(timeout=t)
            except subprocess.TimeoutExpired:
                kill_all(user)
                p.kill()
                try:
                    p.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                return emit({"id": req["id"], "ok": False, "out": "timeout"})
            out = raw.decode(errors="replace")[-2000:]
            if p.returncode != 0:
                return emit({"id": req["id"], "ok": False, "out": ("compile: " if label == "compile" else "") + out})
        emit({"id": req["id"], "ok": True, "out": out})
    except Exception as e:
        emit({"id": req["id"], "ok": False, "out": f"worker error: {type(e).__name__}: {e}"})
    finally:
        cleanup(user, d)
        os.close(fd)
pool = ThreadPoolExecutor(%(threads)d)
for line in sys.stdin:
    if line.strip():
        pool.submit(handle, json.loads(line))
pool.shutdown(wait=True)
"""


class _Worker:
    """Host side of the persistent worker: thread-safe; requests from several threads are multiplexed by id."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.waiting: dict[int, tuple[threading.Event, dict]] = {}
        self.next_id = 0
        self.broken = os.environ.get("SLMBENCH_SANDBOX_WORKER", "1") == "0"

    def _start(self) -> bool:
        if self.broken:
            return False
        if self.proc and self.proc.poll() is None:
            return True
        code = WORKER % {"tools": repr(SBX_TOOLS), "env": repr(ENV), "threads": min(SLOTS, max(2, default_cpus())),
                         "slots": SLOTS, "jobs": JOBS, "locks": LOCKS, "fsize": FSIZE, "nproc": NPROC}
        try:
            # env -i: the worker and everything it runs see only PATH/HOME/LANG + the Go settings, no sandbox placeholders
            self.proc = subprocess.Popen(["sbx", "exec", "-i", NAME, "env", "-i", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                                          "HOME=/home/agent", "LANG=C.UTF-8", "python3", "-u", "-c", code],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                         env=_sbx_env())
        except OSError:
            self.broken = True
            return False
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        atexit.register(self.close)
        return True

    def _reader(self, proc):
        for line in proc.stdout:
            try:
                res = json.loads(line)
            except ValueError:
                continue
            with self.lock:
                slot = self.waiting.get(res.get("id"))
            if slot:
                slot[1].update(res)
                slot[0].set()
        with self.lock:                                  # the worker exited: wake everybody (they fall back)
            for ev, res in self.waiting.values():
                res.setdefault("dead", True)
                ev.set()

    def run(self, files, steps) -> tuple[bool, str] | None:
        """(ok, output), or None if the worker isn't usable (the caller falls back to one sbx exec per check)."""
        with self.lock:
            if not self._start():
                return None
            self.next_id += 1
            rid = self.next_id
            ev, res = threading.Event(), {}
            self.waiting[rid] = (ev, res)
            req = json.dumps({"id": rid, "tar": base64.b64encode(_tar(files)).decode(),
                              "steps": [[argv, t, label] for argv, t, label in steps]}) + "\n"
            try:
                self.proc.stdin.write(req.encode())
                self.proc.stdin.flush()
            except OSError:
                self.waiting.pop(rid, None)
                self.proc = None
                return None
        budget = sum(t for _, t, _ in steps) + 60
        done = ev.wait(budget)
        with self.lock:
            self.waiting.pop(rid, None)
        if not done or res.get("dead"):
            return None
        return bool(res.get("ok")), res.get("out", "")

    def close(self):
        p = self.proc
        if p and p.poll() is None:
            try:
                p.stdin.close()
                p.wait(timeout=30)
            except Exception:                            # noqa: BLE001
                p.kill()


_worker_singleton: _Worker | None = None


def _worker() -> _Worker:
    global _worker_singleton
    if _worker_singleton is None:
        _worker_singleton = _Worker()
    return _worker_singleton
