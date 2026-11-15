"""Browsing pi.dev through MCP: the model drives a real Chrome with the Chrome DevTools MCP server, using the Pi coding agent
(https://pi.dev) as its harness, on the pi.dev website. `slmbench browse` runs it in two ways:

- **demo** (`slmbench browse ["task"]`): Chrome opens pi.dev in a visible window and Pi's interactive interface takes the
  terminal. The task given (if any) is the first message; then you type more in Pi, and leave it with Ctrl+D. Not scored.
- **scored** (`slmbench browse --tasks FILE`): your task file, one episode per task. Each task says what to do and how to
  check it: the expected answer (in the agent's final reply) and/or the address the browser must end on. Every episode is
  watched (visible Chrome + Pi's interface) unless `--headless` or there is no display or terminal, and the run is saved
  with a report.

```text
 slmbench browse
   ├─ Chrome for Testing, remote debugging on 127.0.0.1, only pi.dev and its subdomains resolve
   │    watched: a visible window that outlines each element the agent acts on; otherwise headless
   └─ per task:  CDP → a fresh tab on the task's start page (https://pi.dev/ by default)
                 pi  (only the Chrome DevTools MCP tools: snapshot, click, fill, type, key, hover, navigate)
                    │  watched: Pi's interactive interface, with the pi_watch.ts extension writing the agent's events to a
                    │  file; headless: `pi -p --mode json` (the same events on stdout)
                    └─ chrome-devtools-mcp --browserUrl …  ── acts on the same tab
                 the agent's final reply and the tab's address → the task's checks
```

The model never gets a shell or file tools (Pi's built-in tools are off), nor JavaScript evaluation through the MCP server:
the only way to do a task is to use the website like a person would. Everything is installed by `slmbench browse --setup`
(or on first use) into slmbench's data folder, at pinned versions: Pi, Chrome DevTools MCP and Chrome for Testing.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PI = ("@earendil-works/pi-coding-agent", "1.5.2")
MCP = ("chrome-devtools-mcp", "1.13.0")
BROWSERS = ("@puppeteer/browsers", "3.2.7")
CHROME = "chrome@157.0.8103.78"                 # Chrome for Testing build
NODE_MIN = (22, 19)                              # Pi needs Node.js ≥ 22.19

SITE = "pi.dev"                                  # the only site the browser can reach (and its subdomains: docs.pi.dev…)
START_URL = "https://pi.dev/"
SERVER = "chrome-devtools"                       # MCP server name: its tools reach the model as mcp__chrome_devtools__<tool>
TOOLS = ["take_snapshot", "click", "fill", "type_text", "press_key", "hover", "navigate_page"]
MCP_ATTEMPTS = 3                                 # tries per episode when the MCP tools aren't ready in time
MAX_STEPS = 30                                   # tool calls per episode
EPISODE_TIMEOUT = 300                            # seconds per episode, at most
EXTENSION = Path(__file__).with_name("pi_watch.ts")   # reports Pi's events to slmbench when its interface is on screen
WATCH_LINGER = 2.0                               # watched: seconds the agent's last answer stays on screen

SYSTEM_PROMPT = (
    "You control a web browser through Chrome DevTools tools. The browser shows the website https://pi.dev (Pi, a coding "
    "agent); only pi.dev and its subdomains, such as docs.pi.dev, can be opened. Do what the user asks using that website. "
    "Call take_snapshot to see the page: every element has a uid. Act with click, fill, type_text, press_key or hover using a "
    "uid from the latest snapshot, or navigate_page to go back or to open a pi.dev address, and take a new snapshot after the "
    "page changes. When you are done, reply with the answer in one short sentence."
)

# Watched: outlines the element the agent acts on (click, focus, typing) for a moment. It only changes the element's inline
# outline, never the page's structure or its accessibility tree, so what the agent sees is the same.
HIGHLIGHT_JS = """(() => {
  if (window.__slmbenchHighlight) return true;
  window.__slmbenchHighlight = true;
  const flash = (el, color) => {
    if (!(el instanceof HTMLElement) || el === document.body || el === document.documentElement) return;
    const old = [el.style.outline, el.style.outlineOffset];
    el.style.outline = '3px solid ' + color;
    el.style.outlineOffset = '2px';
    clearTimeout(el.__slmbenchT);
    el.__slmbenchT = setTimeout(() => { el.style.outline = old[0]; el.style.outlineOffset = old[1]; }, 1200);
  };
  document.addEventListener('pointerdown', e => flash(e.target, '#e11d48'), true);
  document.addEventListener('focusin', e => flash(e.target, '#2563eb'), true);
  document.addEventListener('input', e => flash(e.target, '#16a34a'), true);
  return true;
})()"""


def tool_names() -> list[str]:
    return [f"mcp__{SERVER.replace('-', '_')}__{t}" for t in TOOLS]


def on_site(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == SITE or host.endswith("." + SITE)


def watch_available() -> bool:
    """Watching needs a terminal (Pi's interface) and a display (Chrome's window)."""
    try:
        tty = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        tty = False
    display = sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return tty and display


# ---------------------------------------------------------------------------------------------
# Installation (slmbench browse --setup)
# ---------------------------------------------------------------------------------------------
def root() -> Path:
    from .engine import data_dir
    return data_dir() / "browser"


def _bin(name: str) -> Path:
    return root() / "node_modules" / ".bin" / name


def node_version() -> tuple[int, ...] | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
        return tuple(int(x) for x in out.lstrip("v").split(".")[:2])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def chrome_path() -> Path | None:
    try:
        p = Path(json.loads((root() / "chrome.json").read_text())["path"])
        return p if p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def _installed(pkg: str) -> str | None:
    try:
        return json.loads((root() / "node_modules" / pkg / "package.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        return None


def status() -> list[str]:
    """What is missing to browse (empty = ready)."""
    miss = []
    v = node_version()
    if v is None:
        miss.append("Node.js ≥ 22.19 (https://nodejs.org)")
    elif v < NODE_MIN:
        miss.append(f"Node.js ≥ 22.19 (found {'.'.join(map(str, v))})")
    for pkg, ver in (PI, MCP):
        if _installed(pkg) != ver:
            miss.append(f"{pkg}@{ver}")
    if not chrome_path():
        miss.append("Chrome for Testing")
    return miss


def provision(on_line=print) -> bool:
    """Installs Pi, Chrome DevTools MCP and Chrome for Testing (pinned versions) in slmbench's data folder. Nothing is
    installed globally."""
    r = root()
    r.mkdir(parents=True, exist_ok=True)
    v = node_version()
    if v is None or v < NODE_MIN:
        on_line("✘ " + ("Node.js ≥ 22.19 is required (https://nodejs.org)" if v is None else
                       f"Node.js ≥ 22.19 is required (found {'.'.join(map(str, v))})"))
        return False
    npm = shutil.which("npm")
    if not npm:
        on_line("✘ npm not found (it comes with Node.js)")
        return False
    if not (r / "package.json").exists():
        (r / "package.json").write_text(json.dumps({"name": "slmbench-browser", "private": True}, indent=1))
    want = [f"{p}@{v}" for p, v in (PI, MCP, BROWSERS)]
    if all(_installed(p) == v for p, v in (PI, MCP, BROWSERS)):
        on_line("✔ " + ", ".join(want))
    else:
        on_line("npm install " + " ".join(want) + f"  ({r})")
        p = subprocess.run([npm, "install", "--no-fund", "--no-audit", "--save-exact", *want], cwd=str(r), capture_output=True, text=True)
        if p.returncode != 0:
            on_line("✘ npm: " + (p.stderr.strip().splitlines() or ["?"])[-1][:300])
            return False
        on_line("✔ " + ", ".join(want))
    if chrome_path():
        on_line(f"✔ {CHROME} ({chrome_path()})")
    else:
        on_line(f"{CHROME} …")
        p = subprocess.run([str(_bin("browsers")), "install", CHROME, "--path", str(r / "chrome")], cwd=str(r), capture_output=True, text=True)
        line = next((l for l in reversed(p.stdout.strip().splitlines()) if l.startswith("chrome@")), "")
        path = line.split(" ", 1)[1].strip() if " " in line else ""
        if p.returncode != 0 or not path or not Path(path).exists():
            on_line("✘ Chrome: " + ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1][:300])
            return False
        (r / "chrome.json").write_text(json.dumps({"version": CHROME, "path": path}, indent=1))
        on_line(f"✔ {CHROME} ({path})")
    return not status()


# ---------------------------------------------------------------------------------------------
# Chrome and the Chrome DevTools Protocol
# ---------------------------------------------------------------------------------------------
class Chrome:
    """Chrome with remote debugging on 127.0.0.1 and a throwaway profile, headless or in a visible window. Only pi.dev and
    its subdomains resolve: a link to any other site (GitHub, Discord…) doesn't load."""

    def __init__(self, path: Path, visible: bool = False):
        self.profile = Path(tempfile.mkdtemp(prefix="slmbench-chrome-"))
        view = ["--window-size=1100,900", "--window-position=40,40"] if visible else ["--headless=new", "--disable-gpu",
                                                                                       "--window-size=1280,900"]
        args = [str(path), *view, "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
                f"--user-data-dir={self.profile}", "--no-first-run", "--no-default-browser-check",
                "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-sync",
                "--disable-default-apps", "--mute-audio", "--disable-infobars",
                # never the system keyring (GNOME Keyring / KWallet would ask for a password, and Chrome waits for it before
                # loading pages): the throwaway profile keeps its own, unencrypted, store
                "--password-store=basic", "--use-mock-keychain",
                f"--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE {SITE}, EXCLUDE *.{SITE}", "about:blank"]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        port_file = self.profile / "DevToolsActivePort"
        end = time.time() + 30
        while time.time() < end and self.proc.poll() is None:
            try:
                self.port = int(port_file.read_text().split()[0])
                break
            except (OSError, ValueError, IndexError):
                time.sleep(0.1)
        else:
            self.close()
            raise RuntimeError("Chrome did not start (no DevToolsActivePort)")
        self.url = f"http://127.0.0.1:{self.port}"

    def _json(self, path: str, method: str = "GET"):
        req = urllib.request.Request(self.url + path, method=method)
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode()
        try:
            return json.loads(body)
        except ValueError:
            return body

    def fresh_tab(self, url: str) -> str:
        """Opens url in a new tab and closes every other one (no history from the previous task). Returns its WebSocket."""
        old = [t for t in self._json("/json/list") if t.get("type") == "page"]
        tab = self._json("/json/new?" + url, method="PUT")
        for t in old:
            try:
                self._json(f"/json/close/{t['id']}")
            except Exception:  # noqa: BLE001
                pass
        return tab["webSocketDebuggerUrl"]

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.SubprocessError:
            pass
        shutil.rmtree(self.profile, ignore_errors=True)


async def _cdp(ws_url: str, calls: list[tuple[str, dict]], timeout: float = 20) -> list[dict]:
    import aiohttp
    out = []
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(ws_url, max_msg_size=0, timeout=timeout) as ws:
            for i, (method, params) in enumerate(calls, 1):
                await ws.send_json({"id": i, "method": method, "params": params})
                while True:
                    r = await asyncio.wait_for(ws.receive_json(), timeout)
                    if r.get("id") == i:
                        if "error" in r:
                            raise RuntimeError(f"CDP {method}: {r['error'].get('message')}")
                        out.append(r.get("result") or {})
                        break
    return out


def evaluate(ws_url: str, expr: str):
    r = asyncio.run(_cdp(ws_url, [("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})]))[0]
    if r.get("exceptionDetails"):
        raise RuntimeError((r["exceptionDetails"].get("exception") or {}).get("description") or r["exceptionDetails"].get("text"))
    return (r.get("result") or {}).get("value")


def open_start(chrome: Chrome, url: str, highlight: bool) -> str:
    """A fresh tab on the start page, loaded. Returns its WebSocket."""
    ws = chrome.fresh_tab(url)
    end = time.time() + 45
    while time.time() < end:
        try:
            if evaluate(ws, "document.readyState") == "complete" and evaluate(ws, "location.href").startswith("http"):
                break
        except Exception:  # noqa: BLE001 - still loading
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError(f"{url} did not load (is there internet access?)")
    if highlight:
        mark(ws)
    return ws


def mark(ws: str):
    """(Re)installs the highlight on the tab's current page (it is lost when the agent opens another page)."""
    try:
        evaluate(ws, HIGHLIGHT_JS)
    except Exception:  # noqa: BLE001 - the page is changing: the next action installs it again
        pass


def current_url(ws: str) -> str:
    try:
        return evaluate(ws, "location.href") or ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------------------------
# Pi
# ---------------------------------------------------------------------------------------------
def write_pi_config(agent_dir: Path, *, base_url: str, model: str, api_key: bool, chrome_url: str, max_tokens: int,
                    context: int, temperature: float, sampling: dict) -> None:
    """Pi's own folder for this run (PI_CODING_AGENT_DIR): the model as an OpenAI-compatible provider and the Chrome DevTools
    MCP server with only the browser tools exposed. The user's ~/.pi is never read or changed."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    models = {"providers": {"slmbench": {
        "name": "slmbench", "baseUrl": base_url, "api": "openai-completions",
        "apiKey": "$SLMBENCH_PI_API_KEY" if api_key else "none",
        "models": [{"id": model, "name": model, "reasoning": False, "input": ["text"], "contextWindow": int(context),
                    "maxTokens": int(max_tokens), "samplingParams": {"temperature": temperature, **sampling}}]}}}
    args = ["--browserUrl", chrome_url, "--no-page-id-routing", "--no-usage-statistics", "--no-performance-crux",
            "--no-javascript-evaluation"]
    mcp = {"autoEnableCodemode": False, "mcpServers": {SERVER: {
        "command": str(_bin("chrome-devtools-mcp")), "args": args, "env": {"CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1"},
        "exposure": "hidden", "toolExposure": {t: "direct" for t in TOOLS}, "timeout": 60}}}
    (agent_dir / "models.json").write_text(json.dumps(models, indent=1))
    (agent_dir / "mcp.json").write_text(json.dumps(mcp, indent=1))


def pi_command(model: str, first: str | None, watch: bool) -> list[str]:
    """Pi with only the browser tools. watch: its interactive interface (first = its first message, if any), with the
    extension that reports its events; otherwise print mode with the events as JSON on stdout."""
    mode = ["-e", str(EXTENSION)] if watch else ["-p", "--mode", "json"]
    return [str(_bin("pi")), *mode, "--no-session", "--no-context-files", "--no-skills", "--no-prompt-templates",
            "--tools", ",".join(tool_names()), "--model", f"slmbench/{model}", "--system-prompt", SYSTEM_PROMPT,
            *([first] if first else [])]


def _kill(p: subprocess.Popen):
    if p.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except OSError:
            return
        try:
            p.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue


def _tty_state():
    """The terminal's settings, to put back if Pi had to be killed while its interface was on screen."""
    try:
        import termios
        return termios.tcgetattr(sys.stdin.fileno())
    except Exception:  # noqa: BLE001 - not a terminal, or not POSIX
        return None


def _tty_flush():
    """Drops keys typed while no agent was on screen, so they never reach the next task's Pi."""
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:  # noqa: BLE001
        pass


def _tty_restore(saved, hard: bool):
    if saved is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
    except Exception:  # noqa: BLE001
        pass
    if hard:   # killed: leave Pi's alternate screen and give back the cursor, mouse and keyboard modes
        sys.stdout.write("\x1b[?1049l\x1b[?25h\x1b[?2004l\x1b[?1000l\x1b[?1002l\x1b[?1006l\x1b[<u\n")
        sys.stdout.flush()


def _follow(path: Path, p: subprocess.Popen, events: queue.Queue):
    """Puts each line the extension appends to `path` in `events` until Pi exits, then None."""
    pos, buf = 0, b""
    while True:
        alive = p.poll() is None
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        pos += len(chunk)
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            if line.strip():
                events.put(line)
        if not alive:
            break
        time.sleep(0.1)
    events.put(None)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n… ({len(s):,} chars) …\n" + s[-n // 2:]


def _slim(msg: dict) -> dict:
    """A message of the transcript, kept small enough for episodes.jsonl (snapshots can be long)."""
    out = {"role": msg.get("role")}
    content = msg.get("content")
    if isinstance(content, list):
        parts = []
        for c in content:
            t = c.get("type")
            if t == "text":
                parts.append({"type": "text", "text": _trim(c.get("text", ""), 6000)})
            elif t == "thinking":
                parts.append({"type": "thinking", "thinking": _trim(c.get("thinking", ""), 3000)})
            elif t == "toolCall":
                parts.append({"type": "toolCall", "name": c.get("name"), "arguments": c.get("arguments")})
        out["content"] = parts
    else:
        out["content"] = _trim(str(content or ""), 6000)
    for k in ("toolName", "isError", "stopReason", "errorMessage"):
        if msg.get(k) is not None:
            out[k] = msg[k]
    return out


# ---------------------------------------------------------------------------------------------
# Model settings → Pi
# ---------------------------------------------------------------------------------------------
class Session:
    """Chrome + Pi's configuration for one `slmbench browse`: the model from slmbench's settings (server, temperature, max
    tokens, reasoning budget, extra body, context), the same as the other modes."""

    def __init__(self, s, model: str, workdir: Path, visible: bool):
        from .engine import server_kind
        self.model, self.visible = model, visible
        sampling = {}
        if s.reasoning_budget >= 0 and server_kind(s.base_url, None) != "api":
            sampling["thinking_budget_tokens"] = int(s.reasoning_budget)
        try:
            sampling.update(json.loads(s.extra_body or "{}"))
        except ValueError:
            pass
        self.chrome = Chrome(chrome_path(), visible=visible)
        self.agent_dir = workdir / "pi-agent"
        write_pi_config(self.agent_dir, base_url=s.base_url, model=model, api_key=bool(s.api_key), chrome_url=self.chrome.url,
                        max_tokens=int(s.ld_max_new_tokens or 2048), context=int(s.max_context or 32768),
                        temperature=float(s.temperature or 0), sampling=sampling)
        self.env = dict(os.environ, PI_CODING_AGENT_DIR=str(self.agent_dir), PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1",
                        PI_TELEMETRY="0", CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS="1", SLMBENCH_PI_API_KEY=s.api_key or "")
        self.api_key = s.api_key or ""

    def close(self):
        self.chrome.close()


# ---------------------------------------------------------------------------------------------
# Demo: Pi's interface on pi.dev, nothing scored
# ---------------------------------------------------------------------------------------------
def demo(s, model: str, task: str | None = None) -> int:
    """Chrome shows pi.dev and Pi's interface takes the terminal; `task` (if any) is the first message. Returns Pi's code."""
    work = Path(tempfile.mkdtemp(prefix="slmbench-browse-"))
    sess = None
    try:
        sess = Session(s, model, work, visible=True)
        ws = open_start(sess.chrome, START_URL, highlight=True)
        log = work / "events.jsonl"
        env = dict(sess.env, SLMBENCH_PI_EVENTS=str(log), SLMBENCH_PI_LABEL=f"slmbench · {SITE}", SLMBENCH_PI_EXIT_ON_SETTLE="0")
        tty = _tty_state()
        _tty_flush()
        cwd = work / "cwd"
        cwd.mkdir()
        p = subprocess.Popen(pi_command(model, task, watch=True), cwd=str(cwd), env=env, start_new_session=True)
        events: queue.Queue = queue.Queue()
        threading.Thread(target=_follow, args=(log, p, events), daemon=True).start()
        try:
            while (line := events.get()) is not None:      # keep the highlight on every page the agent opens
                if b'"tool_execution_end"' in line:
                    mark(ws)
        except KeyboardInterrupt:
            pass
        finally:
            _kill(p)
            _tty_restore(tty, hard=p.returncode == -signal.SIGKILL)
        return p.returncode or 0
    finally:
        if sess:
            sess.close()
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Scored tasks (your task file)
# ---------------------------------------------------------------------------------------------
def load_tasks(path: str | Path) -> list[dict]:
    """A JSON list (or JSON Lines) of tasks: {"task": what to do, "answer": text (or list of texts, any of them) the final
    reply must contain, "url": text the final address must contain, "id": a name, "start": a pi.dev address to start on}.
    "task" and at least one of "answer" / "url" are required."""
    raw = Path(path).expanduser().read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except ValueError:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(data, dict):
        data = data.get("tasks") or []
    tasks = []
    for i, t in enumerate(data, 1):
        if not isinstance(t, dict) or not str(t.get("task") or "").strip():
            raise ValueError(f"task {i}: needs a \"task\" text")
        if not t.get("answer") and not t.get("url"):
            raise ValueError(f"task {i}: needs an \"answer\" and/or a \"url\" to check it")
        start = t.get("start") or START_URL
        if not on_site(start):
            raise ValueError(f"task {i}: \"start\" must be a {SITE} address")
        answers = t.get("answer") or []
        tasks.append({"id": str(t.get("id") or i), "task": str(t["task"]).strip(), "start": start,
                      "answer": [str(a) for a in (answers if isinstance(answers, list) else [answers])],
                      "url": str(t["url"]) if t.get("url") else None})
    if not tasks:
        raise ValueError("no tasks in the file")
    return tasks


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w.@/:+-]+", " ", s.lower()).split())


def check(task: dict, final: str, url: str) -> tuple[bool, str]:
    """(success, why): the reply contains one of the expected answers (case, spaces and punctuation ignored), and the
    final address contains the expected one."""
    why = []
    if task["answer"]:
        if not any(_norm(a) and _norm(a) in _norm(final) for a in task["answer"]):
            why.append("answer not in the reply")
    if task.get("url"):
        if task["url"].lower().rstrip("/") not in url.lower().rstrip("/") + "/":
            why.append(f"ended on {url or '?'}")
    return not why, "; ".join(why)


class _Stop(Exception):
    pass


def _next(events: queue.Queue, timeout: float = 3):
    try:
        return events.get(timeout=timeout)
    except queue.Empty:
        return None


def run_task(sess: Session, task: dict, *, max_steps: int, timeout: float, watch: bool, on_call=None) -> dict:
    """One episode. Pi waits at most 10 s for the MCP server before its first request; on a loaded machine the model could
    get no tools. Pi's system message lists the tools it declares (`toolsAdded`): if the browser tools are missing, the
    attempt is dropped before any model call and the task starts again."""
    t0 = time.time()
    for attempt in range(1, MCP_ATTEMPTS + 1):
        ep = _play(sess, task, max_steps=max_steps, timeout=timeout, watch=watch, on_call=on_call)
        if ep["stop"] != "no_tools":
            break
    ep["seconds"] = round(time.time() - t0, 1)
    if attempt > 1:
        ep["attempts"] = attempt
    if ep["stop"] == "no_tools":
        ep["stop"] = "setup_error"
        ep["error"] = f"the Chrome DevTools MCP tools were not ready after {MCP_ATTEMPTS} attempts (Pi waits 10 s for them)"
    return ep


def _play(sess: Session, task: dict, *, max_steps: int, timeout: float, watch: bool, on_call=None) -> dict:
    t0 = time.time()
    ep = {"id": task["id"], "task": task["task"]}
    try:
        ws = open_start(sess.chrome, task["start"], highlight=watch)
    except Exception as e:  # noqa: BLE001
        return {**ep, "success": 0, "why": "setup error", "url": "", "steps": 0, "tool_errors": 0, "turns": 0,
                "input_tokens": 0, "output_tokens": 0, "seconds": round(time.time() - t0, 1), "stop": "setup_error",
                "error": f"{type(e).__name__}: {e}", "final": "", "messages": [], "calls": []}
    work = Path(tempfile.mkdtemp(prefix="slmbench-pi-"))       # empty cwd: no project files or .pi/ settings for Pi to pick up
    events: queue.Queue = queue.Queue()
    err_tail: list[str] = []
    tty = None
    if watch:
        log = work / "events.jsonl"
        env = dict(sess.env, SLMBENCH_PI_EVENTS=str(log), SLMBENCH_PI_LABEL=f"slmbench · task {task['id']}",
                   SLMBENCH_PI_LINGER_MS=str(int(WATCH_LINGER * 1000)), SLMBENCH_PI_EXIT_ON_SETTLE="1")
        tty = _tty_state()
        _tty_flush()
        p = subprocess.Popen(pi_command(sess.model, task["task"], watch=True), cwd=str(work), env=env, start_new_session=True)
        threading.Thread(target=_follow, args=(log, p, events), daemon=True).start()
    else:
        p = subprocess.Popen(pi_command(sess.model, task["task"], watch=False), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=str(work), env=dict(sess.env, NO_COLOR="1"), start_new_session=True)

        def read_out():
            buf = b""
            while chunk := p.stdout.read1(65536):
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if line.strip():
                        events.put(line)
            events.put(None)

        def read_err():
            for line in p.stderr:
                err_tail.append(line.decode(errors="replace").rstrip())
                del err_tail[:-40]

        threading.Thread(target=read_out, daemon=True).start()
        threading.Thread(target=read_err, daemon=True).start()
    messages, calls, declared = [], [], set()
    steps = tool_errors = turns = tin = tout = retried = 0
    stop, error, msg_t0, settled, ended = "answered", None, None, False, False
    deadline = t0 + timeout
    try:
        while True:
            left = deadline - time.time()
            if left <= 0:
                stop = "timeout"
                break
            try:
                line = events.get(timeout=min(1.0, left))
            except queue.Empty:
                continue
            if line is None:
                ended = True
                if not settled:
                    # watched: Pi closed by the viewer (Ctrl+C twice, Ctrl+D) before the agent finished
                    stop = "viewer_cancelled" if watch and p.returncode in (0, 130) else "pi_exited"
                break
            try:
                e = json.loads(line)
            except ValueError:
                continue
            kind = e.get("type")
            if kind == "message_start" and (e.get("message") or {}).get("role") == "assistant":
                msg_t0 = time.time()
            elif kind == "message_end":
                m = e.get("message") or {}
                if m.get("role") == "system":            # lists the tools Pi declares to the model
                    declared |= {t.get("name") for t in m.get("toolsAdded") or [] if isinstance(t, dict)}
                    continue
                if m.get("role") == "user" and turns == 0 and not set(tool_names()) <= declared:
                    stop = "no_tools"                        # the MCP server isn't connected yet: retry before any model call
                    raise _Stop
                messages.append(_slim(m))
                if m.get("role") == "assistant":
                    turns += 1
                    u = m.get("usage") or {}
                    tin, tout = tin + (u.get("input") or 0), tout + (u.get("output") or 0)
                    if on_call:
                        on_call({"t": round(time.time() - (msg_t0 or time.time()), 3), "ct": u.get("output")})
                    if m.get("stopReason") == "error":         # Pi retries failed requests by itself: only the last one counts
                        error = m.get("errorMessage") or "model request failed"
                        retried += 1
                    else:
                        error = None
            elif kind == "tool_execution_start":
                steps += 1
                calls.append({"tool": str(e.get("toolName", "")).split("__")[-1], "args": e.get("args")})
                if steps > max_steps:
                    stop = "max_steps"
                    raise _Stop
            elif kind == "tool_execution_end":
                if e.get("isError"):
                    tool_errors += 1
                    if calls:
                        calls[-1]["error"] = _trim(_text((e.get("result") or {}).get("content")), 400)
                if watch:
                    mark(ws)
            elif kind == "agent_settled":
                settled = True
                if e.get("aborted"):           # watched: the viewer pressed Escape
                    stop = "viewer_cancelled"
                break
    except _Stop:
        pass
    finally:
        url = current_url(ws)
        if watch:
            if settled:                        # the extension closes Pi after showing the agent's last answer
                try:
                    p.wait(timeout=WATCH_LINGER + 5)
                except subprocess.TimeoutExpired:
                    pass
            elif p.poll() is None and not isinstance(sys.exc_info()[1], KeyboardInterrupt):
                time.sleep(WATCH_LINGER)       # stopped by slmbench (limit): show where the agent was for a moment
        _kill(p)
        if watch:
            _tty_restore(tty, hard=p.returncode == -signal.SIGKILL)
            while not ended and (line := _next(events)) is not None:   # Escape pressed while the answer was on screen
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "agent_settled" and e.get("aborted") and stop != "viewer_cancelled":
                    ep["cancel_run"] = True
        shutil.rmtree(work, ignore_errors=True)
    if stop == "pi_exited" and not error:
        error = "pi exited with code %s: %s" % (p.returncode, " | ".join(err_tail[-3:])[:400])
    final = next((_text(m["content"]) for m in reversed(messages) if m["role"] == "assistant" and _text(m["content"]).strip()), "")
    ok, why = check(task, final, url) if stop == "answered" and not error else (False, stop if not error else "model error")
    return {**ep, "success": int(ok), "why": why, "url": url, "steps": steps, "tool_errors": tool_errors, "turns": turns,
            "input_tokens": tin, "output_tokens": tout, "seconds": round(time.time() - t0, 1), "stop": stop, "error": error,
            "final": final, "messages": messages, "calls": calls, "request_errors": retried}


def _mean_sem(xs: list[float]) -> tuple[float, float | None]:
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else None)


def results(eps: list[dict]) -> dict:
    if not eps:
        return {}
    sr, sr_se = _mean_sem([e["success"] for e in eps])
    calls = sum(e["steps"] for e in eps)
    stops: dict[str, int] = {}
    for e in eps:
        stops[e["stop"]] = stops.get(e["stop"], 0) + 1
    return {"success_rate": sr, "stderr": sr_se, "successes": sum(e["success"] for e in eps), "episodes": len(eps),
            "tool_error_rate": (sum(e["tool_errors"] for e in eps) / calls) if calls else 0.0,
            "mean_steps": round(calls / len(eps), 1), "mean_seconds": round(sum(e["seconds"] for e in eps) / len(eps), 1),
            "stops": stops, "retried_requests": sum(e.get("request_errors") or 0 for e in eps)}


def report(meta: dict, res: dict, eps: list[dict]) -> str:
    cell = lambda s: " ".join(str(s or "").split()).replace("|", "\\|")[:140]  # noqa: E731
    lines = [f"# {SITE} · {meta['model']}", "",
             f"- Tasks: `{meta['tasks_file']}` · {res.get('episodes', 0)} of {meta['n_tasks']} played"
             + (" (cancelled)" if meta.get("cancelled") else ""),
             f"- Harness: Pi {PI[1]} · Chrome DevTools MCP {MCP[1]} · {CHROME} · tools: {', '.join(TOOLS)}",
             f"- Model: `{meta['model']}` · {meta['base_url']} · temperature {meta['temperature']} · max tokens "
             f"{meta['max_tokens']} · reasoning ≤ {meta['reasoning_budget']}",
             f"- Date: {meta['date']} · {'watched' if meta['watch'] else 'headless'} · limits: {MAX_STEPS} actions, "
             f"{meta['timeout']:.0f} s per task", ""]
    if res:
        se = f" ± {res['stderr'] * 100:.1f}" if res.get("stderr") is not None else ""
        lines += [f"**Success: {res['success_rate'] * 100:.1f}%{se}** ({res['successes']}/{res['episodes']}) · "
                  f"{res['mean_steps']} actions and {res['mean_seconds']} s per task · tool errors "
                  f"{res['tool_error_rate'] * 100:.1f}% · how tasks ended: "
                  + ", ".join(f"{k} {v}" for k, v in res["stops"].items()), "",
                  "| # | Task | ✔ | Actions | s | Reply | Why it failed |", "|---|---|:-:|--:|--:|---|---|"]
        for e in eps:
            lines.append(f"| {cell(e['id'])} | {cell(e['task'])} | {'✔' if e['success'] else '✘'} | {e['steps']} | "
                         f"{e['seconds']:.0f} | {cell(e['final'])} | {cell(e['why'] if not e['success'] else '')} |")
    lines += ["", "Every episode (transcript, tool calls, final address) is in `episodes.jsonl`."]
    return "\n".join(lines) + "\n"


def run_tasks(s, model: str, tasks_file: str | Path, *, watch: bool, on_line=print, max_steps: int = MAX_STEPS) -> Path:
    """Plays every task of the file and saves the run (tasks.json, episodes.jsonl, results.json, report.md) in a folder of
    the results directory. Escape in Pi (watched) or Ctrl+C stops it; finished tasks are kept."""
    tasks = load_tasks(tasks_file)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])
    out = Path(s.out_dir).expanduser() / f"{label}-{SITE.replace('.', '')}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1, ensure_ascii=False))
    timeout = min(float(s.timeout or EPISODE_TIMEOUT), EPISODE_TIMEOUT)
    meta = {"model": model, "base_url": s.base_url, "tasks_file": str(tasks_file), "n_tasks": len(tasks),
            "temperature": s.temperature, "max_tokens": s.ld_max_new_tokens, "reasoning_budget": s.reasoning_budget,
            "date": dt.datetime.now().isoformat(timespec="seconds"), "watch": watch, "timeout": timeout, "cancelled": False}
    on_line(f"{SITE}: {len(tasks)} tasks · Pi {PI[1]} + Chrome DevTools MCP {MCP[1]} · model {model}"
            + (" · watched (Esc in Pi stops the run)" if watch else " · headless"))
    eps: list[dict] = []
    sess = None

    def on_term(signum, frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, on_term)
    try:
        sess = Session(s, model, out, visible=watch)
        for i, task in enumerate(tasks, 1):
            if watch:
                on_line(f"[{i}/{len(tasks)}] {task['id']}: {task['task'][:100]} …")
            ep = run_task(sess, task, max_steps=max_steps, timeout=timeout, watch=watch)
            if ep["stop"] == "viewer_cancelled":         # Escape / Pi closed: the run stops, this task isn't kept
                raise KeyboardInterrupt
            cancel_run = ep.pop("cancel_run", False)
            eps.append(ep)
            line = json.dumps(ep, ensure_ascii=False)
            if sess.api_key:                             # provider error messages are saved with the episode: scrub the key
                line = line.replace(sess.api_key, "***")
            with open(out / "episodes.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            on_line(f"[{i}/{len(tasks)}] {'✔' if ep['success'] else '✘'} {task['id']}: {ep['steps']} actions, "
                    f"{ep['seconds']:.0f}s · {' '.join(ep['final'].split())[:120]}"
                    + ("" if ep["success"] else f"  ({ep['why']})"))
            if cancel_run:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        meta["cancelled"] = True
        on_line("Cancelled.")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        if sess:
            sess.close()
    res = results(eps)
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": res}, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report(meta, res, eps))
    if res:
        on_line(f"{SITE}: {res['success_rate'] * 100:.1f}% ({res['successes']}/{res['episodes']}) · saved to {out}")
    return out
