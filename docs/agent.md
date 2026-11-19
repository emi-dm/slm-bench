# Agent mode: SWE-bench Lite with mini-SWE-agent, Pi or OpenCode

The `agent` mode measures the model **as a coding agent**: it gets a real GitHub issue and the repository at the commit
before the fix, and works on it on its own (it reads code, runs commands, edits files) until it decides it is done. Its
patch is then graded with the **official SWE-bench harness**.

```bash
slmbench setup --mode agent                      # once: the sandbox template, SWE-bench Lite and the image list
slmbench run --mode agent --max-samples 5        # 5 tasks with both harnesses
slmbench run --benchmarks swebench_lite --max-samples 20 --agent-workers 2
slmbench traj <run>                              # every step of the agent (also: Results → e in the TUI)
```

## The benchmark

[SWE-bench Lite](https://www.swebench.com/lite.html): 300 issues from 12 Python projects (django, sympy, scikit-learn,
matplotlib, pytest, sphinx, requests, astropy, xarray, pylint, seaborn, flask), selected from SWE-bench to be
self-contained and solvable from the issue text. slmbench uses the current dataset,
[`SWE-bench/SWE-bench_Lite`](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) (test split), pinned to the
revision downloaded by `setup` (the report says which one).

A task is **resolved** when, after applying the agent's patch, the tests that the real fix makes pass now pass
(`FAIL_TO_PASS`) and every test that passed before still passes (`PASS_TO_PASS`). The test files are reset and the
hidden tests (`test_patch`) are added before running them, so changing the tests does not help.

**Which tasks.** With `--max-samples N` (or *Tasks* in the TUI) the first N tasks of a fixed order are used: the dataset is
sorted by repository, so slmbench orders the tasks by a hash of their id. The first N are the same on every machine and
every run and mix the projects, so two partial runs of the same size are comparable. 0 = every task.

## Two harnesses

| Benchmark | Harness | How the model acts |
|---|---|---|
| `swebench_lite` | [mini-SWE-agent](https://mini-swe-agent.com) 2.4.6 | **bash only**: each answer has one command in a text block (`swebench_backticks.yaml`, the text-based config of its SWE-bench runner). No tool calls, so it does not depend on the server's tool-calling support. Ends when the agent runs the submit command. |
| `swebench_lite_pi` | [Pi](https://pi.dev) 1.5.2 | **native tool calls**: Pi's `read`, `write`, `edit` and `bash` tools, its own system prompt. Ends when the agent answers without a tool call. |
| `swebench_lite_opencode` | [OpenCode](https://opencode.ai) 1.19.5 | **native tool calls**: OpenCode's `build` agent with its own tools (bash, read, edit, write, grep, glob, list, patch, todo…) and system prompt, every permission granted except the web ones. Ends when the agent answers without a tool call. |

All run with the same model settings: temperature, *max tokens per answer* (`agent_max_tokens`, 16384 by default: a
step cut by the limit has no command and is wasted), the reasoning budget and the extra body. The harnesses are told the
server's real context window (llama-server's per-slot `n_ctx`, else the *context* setting), and an answer may use at most
a quarter of it: a harness that reserves more for the answer than the window holds compacts (Pi) or overflows
(OpenCode) at every step. Pi's and OpenCode's own context compaction stay on, as in normal use. Limits per task: *max
steps* (`agent_step_limit`, 100 model calls) and *time* (`agent_timeout`, 1800 s for the agent; grading comes on top).
Running them on the same tasks compares a bash-only agent with two tool-calling ones on the same model.

**Choosing the harness.** In the TUI (*Run benchmarks*, mode *Agent*) the *Harness* field opens a pop-up: mini-SWE-agent,
Pi, OpenCode, or all of them on the same tasks (three times the time). From the command line, with the benchmark keys:
`slmbench run --benchmarks swebench_lite_opencode --max-samples 5`.

## Where it runs

Nothing the model writes runs on your machine, and the sandbox where the agent works has **no network at all**.

```
host: slmbench ── relay ─────────────────────▶ your model server (OpenAI API)
        │  ▲ sbx exec -i (stdin/stdout)
        ▼  │
sbx  slmbench-run-<pid>-a…   from the slmbench-agent template, --deny-network '**'
   ├─ relay.py on 127.0.0.1: the agents' only way to the model
   ├─ mini-SWE-agent / Pi (one process per task)
   ├─ docker: one container per task (the task's image, --network none): every command of the agent runs there
   └─ grading: the patch in a FRESH container of the image (official eval script, log parser and report)
sbx  slmbench-run-<pid>-f…   pulls the task images (registry only) → docker save | docker load into the run sandbox
```

- **The model.** The agents reach it through a relay: an HTTP endpoint inside the sandbox whose requests travel over
  `sbx exec`'s stdin/stdout to slmbench, which forwards only `POST …/chat/completions` to the configured server, always
  for the model under test (the request's model is replaced). Your API key never enters the sandbox and is dropped on a
  redirect to another host; error messages are scrubbed of it before they are saved.
- **The commands.** mini-SWE-agent runs every command with `docker exec` in the task's container. Pi's tools go there
  too: a small extension (`agent_box/pi_docker.ts`, after Pi's own remote-tools example) gives `read`, `write`, `edit`
  and `bash` operations that act through `docker exec`. These containers run with `--network none`, 4 GB of memory and
  2 CPUs.
- **OpenCode** can't send its tools elsewhere, so OpenCode itself runs **inside the task's container** (its binary is
  copied in, with ripgrep for its search tools, its configuration passed in the environment: the relay as an OpenAI-compatible provider, no model list
  download, no autoupdate, no LSP downloads, no project or Claude Code configs). That container shares the sandbox's network
  namespace (`--network host`) to reach the relay on 127.0.0.1; since the sandbox has no network, the relay is all it
  can reach. Its session is exported at the end (`opencode export`) for the trajectory viewer.
- **The images.** Each task has its own image (the repository at the base commit with its environment). The fetcher
  sandbox pulls it and pipes it into the run sandbox, so the run sandbox never needs the network. A task's image is
  removed once the task is graded (the sandbox's Docker disk holds about 10 GB, so at most 3 tasks run in parallel).
- **Memory.** Without a limit a sandbox may take half of the host's RAM, so slmbench sets one: the run sandbox gets
  2 GB + 4 GB per parallel task (`agent_workers`, 1 by default → 6 GB), each task container 4 GB, the fetcher 2 GB;
  both sandboxes get half of the host's cores (at most 8). Mind your model server's memory on top of that.
- **Grading.** `swebench` 5.0.2 inside the sandbox, with its own patch-apply fallbacks, eval script, log parser and
  `get_eval_report`. Only the container runtime differs from `swebench.harness.run_evaluation`.
- **Both sandboxes are deleted** when the run ends: finished, failed, cancelled (`x` in the live view, `Ctrl+C`), or
  slmbench stopped with SIGTERM or by closing its terminal (SIGHUP). Only a `kill -9` can leave them behind; those are
  removed the next time a run starts or the TUI opens (their owner process is gone; a running slmbench's are never
  touched).
- **The template.** `setup` creates a sandbox, installs mini-SWE-agent, swebench, Node.js 22, Pi, OpenCode and ripgrep in `/opt`
  (PyPI, nodejs.org and npm are allowed by Docker Sandboxes' default policy only while it is built), saves it as
  `slmbench-agent:v4` and removes it.

## Apple Silicon (arm64)

The official SWE-bench images are x86_64 only. On arm64, slmbench uses
[Epoch AI's arm64 builds](https://github.com/epoch-research/SWE-bench) of the same environments
(`ghcr.io/epoch-research/swe-bench.eval.arm64.<task>`). They exist for **252 of the 300** Lite tasks; the other 48
(matplotlib 22, scikit-learn 19, xarray 5, astropy 2) are skipped. `setup` checks which ones exist and the report says
how many tasks were available. Because of that, a run on arm64 is never the full benchmark: the leaderboard marks it as
partial (`n=252`), and it is not directly comparable with published SWE-bench Lite numbers.

On x86_64 the official images (`docker.io/swebench/sweb.eval.x86_64.*`) are used and all 300 tasks are available.

## What a run leaves

Besides `report.md`, `summary.json`, `run.log` and `requests.jsonl` (one line per model call: tokens, speed, finish):

- `answers.jsonl`: every answer of the model as the relay saw it (content, reasoning, tool calls, finish), also the ones
  a harness rejects: mini-SWE-agent drops the answers it can't parse (no bash block, two blocks…) and keeps only its
  feedback, so the trajectory viewer shows them from here, where they happened.
- `tasks.jsonl`: one line per task (resolved, exit status, steps, FAIL_TO_PASS, broken PASS_TO_PASS, times).
- `agent/<harness>/<task>/`: the trajectory (`<task>.traj.json` for mini-SWE-agent, `<task>.pi.jsonl` with Pi's JSON
  events, `<task>.opencode.json` with OpenCode's exported session and `<task>.opencode.jsonl` with its run events),
  `patch.diff`, `preds.json`, the grading `report.json` and `test_output.txt`, and `task.log`.

In the TUI, **9 · Agent traces** lists the agent-mode runs → their tasks (one row per harness: resolved or not, steps,
exit status, cut answers) → every step: the reasoning, the answer, the bash command or tool call with its arguments, and
what came back (`n`/`p` move between steps). `e` on an agent run in *Results* opens the same view, and
`slmbench traj <run> [--task ID] [--harness pi] [--step N]` prints it. While a run is going, the live view (`Ctrl+R`) shows
each harness's progress and the agent's latest steps; `x` there asks the run to stop (it removes its sandboxes).

**Watching the harness live.** `w` (on the run screen, or in the live view) opens the session of the task being solved,
updated as it happens and laid out to fit one screen, so there's nothing to scroll: the earlier steps one line each (the
command or tool call, and how it went: ✔/✘ and the size of its output) and the current step in full (the end of its
reasoning, the answer, the action and its output, or "waiting" while it runs). `e` switches to the whole transcript
(scrollable, following the last step), `t` picks another task when several run in parallel. The
harnesses write their own trajectory only when a task ends, so this view is built from what the relay sees: each answer,
and the outputs of the previous step, which travel in the next request (`answers.jsonl`, entries with `"kind": "obs"`). `u` resumes an interrupted run (finished tasks are kept) and `r`
grades the saved patches again without calling the model.

## Exporting a dataset

Runs can be exported as a dataset, one row per attempt (model × harness × task): the task (the issue), the whole session
in chat format (system, the issue, every answer with its reasoning and tool calls, every output; answers a harness
rejected are kept, flagged `rejected`), the solution (the patch) and whether it is correct (`resolved`), plus the steps,
tokens, exit status, the tests' details and the generation settings. The other modes export the same columns (the
prompt, the answer and its score), so runs of every mode can go in one file.

```bash
slmbench export <run> [<run>…]                   # JSONL in <results>/exports/
slmbench export --mode agent --format parquet    # every agent-mode run, Parquet (zstd)
slmbench export --all -o runs.parquet --format parquet
```

In the TUI: *Results* → `x` → which runs (marked, this one, every agent-mode run, every run) → JSONL or Parquet. The
columns are described in `bench/slmbench/export.py`.

## Expectations

SWE-bench Lite is hard for small models: 1–4B models resolve only the easiest tasks, and the benchmark starts to
separate models around 7–14B (specialised coding/agent models do much better). Look at the trajectories: a model can fail
for reasons that are not about code: answers cut by the token limit, no valid command, loops on the same command, trying to
install packages (there is no network).
