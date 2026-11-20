<p align="center">
  <img src="assets/banner.jpg" alt="slmbench: where small models prove their worth" width="820">
</p>

<p align="center">
  <b>Benchmark the small language models you run locally, on public benchmarks, from a friendly TUI.</b><br>
  Built on <a href="https://github.com/huggingface/lighteval">lighteval</a> · works with llama.cpp, Ollama, LM Studio, vLLM or any OpenAI-compatible server
</p>

<p align="center">
  <a href="docs/agent.md"><img src="https://img.shields.io/badge/agent%20mode-SWE--bench%20Lite-10b981" alt="Agent mode: SWE-bench Lite"></a>
  <a href="https://mini-swe-agent.com"><img src="https://img.shields.io/badge/harness-mini--SWE--agent%202.4.6-3b82f6" alt="Harness: mini-SWE-agent 2.4.6"></a>
  <a href="https://pi.dev"><img src="https://img.shields.io/badge/harness-Pi%201.5.2-ec4899" alt="Harness: Pi 1.5.2"></a>
  <a href="https://opencode.ai"><img src="https://img.shields.io/badge/harness-OpenCode%201.19.5-ff8a3d" alt="Harness: OpenCode 1.19.5"></a>
  <a href="docs/agent.md#where-it-runs"><img src="https://img.shields.io/badge/sandbox-no%20network-8b5cf6" alt="Sandbox without network"></a>
</p>

<p align="center">
  <img src="assets/demo.gif" alt="slmbench demo: check the model's memory with hf-mem, run the tool-calling mode with live progress, read the report, explore a failure and check the leaderboard" width="820"><br>
  <sub>A real session: LFM3-1.7B (Q8_0) on llama-server. Its memory with hf-mem, then tool calling (BFCL), the report, a failure and the leaderboard. <a href="assets/demo.mp4">Watch it as MP4</a> (sharper).</sub>
</p>

**slmbench** measures what **small language models** (≈ 1–14B parameters) can really do on your machine:

- **Only public, established benchmarks** (IFEval, GSM8K, MATH-500, IFStruct, BFCL, RULER, HumanEval+, MBPP+, LiveCodeBench, CRUXEval, MultiPL-E…), with their official prompts and metrics, run by Hugging Face's lighteval.
- **Five modes**: general, coding, tool calling, long context and **agent**: the model as a coding agent on real GitHub issues (SWE-bench Lite), driven by **mini-SWE-agent**, **Pi** or **OpenCode**, in a sandbox without network.
- **Your model, your server**: anything with an OpenAI-compatible API, local or hosted.
- **A TUI** to pick the model, run, follow the progress live, read the reports and explore every failure (and every step of an agent), plus a **[leaderboard](leaderboard.md)** that updates itself after every run.
- **Export as a dataset** (JSONL or Parquet): the task, the whole session, the model's solution and whether it's correct.
- **Will it fit?** The memory a model needs, estimated with [hf-mem](https://github.com/alvarobartt/hf-mem) before you download it.

```text
   your model                               slmbench                          results
 +------------------+   OpenAI API   +-------------------------------+     +------------------+
 | llama-server     | <------------- | TUI / CLI                     | --> | report.md        |
 | Ollama           |    prompts     |  general·code·tools·long·agent |     | leaderboard.md   |
 | LM Studio, vLLM  | -------------> |  lighteval (tasks, metrics)   |     | leaderboard.html |
 +------------------+    answers     +-------------------------------+     +------------------+
```

**Contents** · [Quick start](#quick-start) · [Modes](#modes) · [Agent mode](#agent-mode-swe-bench-lite) · [The TUI](#the-tui) · [Will it fit? (hf-mem)](#will-it-fit-memory-estimates-with-hf-mem) · [Choosing the model](#choosing-the-model-to-evaluate) · [Leaderboard](#leaderboard) · [Cheat sheet](#cheat-sheet) · [Browsing pi.dev](#browsing-pidev-pi--chrome-devtools-mcp) · [Built on lighteval](#built-on-lighteval) · [Tips](#tips) · [Security](#security) · [Docs](#documentation)

## Quick start

You need **Python 3.10+** ([uv](https://docs.astral.sh/uv/) recommended). Optional: **[llama.cpp](https://github.com/ggml-org/llama.cpp)** to start GGUF models from the TUI (`brew install llama.cpp`, or a release on Linux / WSL), and **[Docker Sandboxes](https://docs.docker.com/ai/sandboxes/get-started/)** (`sbx`) for the coding benchmarks, whose code only ever runs inside a sandbox.

```bash
git clone https://github.com/emi-dm/slm-bench && cd slm-bench
uv tool install --editable .      # installs the `slmbench` command (update later with `git pull`)
slmbench                          # opens the TUI
```

Then, in the TUI: **1** pick the model (a local `.gguf` or a running server) → **2** run a mode → **4** read the report → **5** check the leaderboard. Or from the command line:

```bash
slmbench setup --mode general                                  # check requirements, download the mode's datasets
slmbench use lfm3-1.7b --base-url http://127.0.0.1:8080/v1     # the model to evaluate
slmbench run --max-samples 20                                  # run the mode's benchmarks (20 problems each)
slmbench leaderboard                                           # ranking
```

<details>
<summary><b>Installation details</b>: PATH, pip, PyTorch with or without CUDA</summary>

- `uv tool install` puts `slmbench` in `~/.local/bin` (add it to your `PATH` if needed). `uv sync` + `uv run slmbench` works too.
- **Why CPU PyTorch, even on a GPU machine?** lighteval needs PyTorch, but slmbench never runs models with it: the model runs in your server (llama.cpp, vLLM, Ollama…), and that's what uses your GPU. The CPU build is ≈ 200 MB instead of several GB of CUDA libraries and doesn't touch your CUDA setup. With pip on Linux, get the same with `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -e .`.
- **Into an environment that already has PyTorch with CUDA** (for example the same venv as vLLM): just `pip install -e .` from the clone. It accepts any PyTorch 2.x, so it reuses yours.
- To block commits that contain an API key: `git config core.hooksPath .githooks`.

</details>

## Modes

| Mode | What it measures | Benchmarks |
|---|---|---|
| `general` | Everyday use: following instructions, reasoning (also in Spanish), returning structured data | IFEval, GSM8K, MATH-500, IFStruct, MGSM (es) |
| `code` | Writing and reasoning about code in Python and five more languages, compiled and run against their tests | HumanEval+, MBPP+, LiveCodeBench, CRUXEval-O/I, MultiPL-E (JavaScript, C++, Java, Go, Rust) |
| `tools` | Picking the right function and arguments, with the model's native `tool_calls` (as agents use it) | BFCL simple, multiple, parallel, parallel multiple |
| `long` | Finding information hidden in long texts | RULER needle-in-a-haystack at 4k and 8k tokens |
| `agent` | Working as a coding agent on real GitHub issues: explore the repo, edit, run, fix (in a sandbox without network) | SWE-bench Lite with mini-SWE-agent (bash only), Pi or OpenCode (tool calls) · [docs/agent.md](docs/agent.md) |
| `all` | Everything above but `agent` (it runs on its own) | — |

```bash
slmbench setup --mode code        # coding: builds the sandbox template (python, node, g++, javac, go, rustc inside it)
slmbench run --mode code
slmbench setup --mode tools       # BFCL: downloads the original files and prepares them
slmbench run --benchmarks humaneval_plus,ifstruct --max-samples 50     # specific benchmarks
slmbench setup --mode agent       # agent: the sandbox template (mini-SWE-agent, Pi, swebench) and SWE-bench Lite
slmbench run --mode agent --max-samples 5
slmbench traj <run>               # every step of the agent: reasoning, answer, command / tool call, output
slmbench export --mode agent --format parquet   # runs as a dataset: task, session, solution, correct or not
```

What you get from every run:

- **pass@1 and pass@5** for coding and tool calling (5 answers per problem, standard unbiased estimator).
- **Saved answers**: a run can be **rescored** without calling the model and an interrupted one **resumed**, asking only for what's missing.
- **Failure explorer**: what the model answered and *why* it failed (compile error, wrong value, timeout, missing tool call, truncated answer…).
- **Errors and log**: failed requests grouped by the server's message, with hints (API key, rate limit, context too long…).
- **Reasoning-aware**: a configurable reasoning budget; empty and truncated answers are counted and reported.

Every benchmark, with what it measures, its dataset, metric, origin and **a real sample task** (the exact prompt the model receives), is in **[docs/benchmarks.md](docs/benchmarks.md)**, in `slmbench benchmarks <key>` and in the TUI (*3 · Benchmarks*).

## Agent mode (SWE-bench Lite)

The model gets a real GitHub issue and the repository at the commit before the fix, and works on it **as a coding agent**:
it explores, runs commands, edits files and decides when it's done. Its patch is graded with the **official SWE-bench
harness** (resolved = the tests of the real fix pass and none breaks). Pick the harness in a pop-up, or all three on the
same tasks to compare a bash-only agent with two tool-calling ones:

| Harness | How the model acts |
|---|---|
| [mini-SWE-agent](https://mini-swe-agent.com) | bash only: one command per answer, as text (no tool calls needed from the server) |
| [Pi](https://pi.dev) | native tool calls: `read`, `write`, `edit`, `bash` |
| [OpenCode](https://opencode.ai) | native tool calls: its `build` agent and its own tools (bash, read, edit, grep, glob…) |

Everything runs in a Docker Sandbox **with no network at all**: the agent's commands run in the task's container, the
model is reached through a relay that only lets chat completions through, and the task images are piped in from a
separate sandbox. On Apple Silicon, 252 of the 300 tasks have an arm64 image (Epoch AI's builds). Every step is saved and
can be explored in the TUI (**9 · Agent traces**), and runs can be exported as a dataset. Details: **[docs/agent.md](docs/agent.md)**.

| | |
|:-:|:-:|
| <img src="assets/screenshots/agent-harness.png" alt="Picking the harness in a pop-up" width="420"><br>**2 · Run → Agent**: the harness in a pop-up, tasks, steps, tokens, time | <img src="assets/screenshots/agent-live.png" alt="Live view of an agent run" width="420"><br>**Live view**: every step of the agent as it happens |
| <img src="assets/screenshots/agent-traces.png" alt="The steps of a task" width="420"><br>**9 · Agent traces**: every step of a task, with its tokens and exit code | <img src="assets/screenshots/agent-step.png" alt="One step: reasoning, tool call and output" width="420"><br>**One step**: reasoning, answer, command or tool call and its output (`n`/`p`) |
| <img src="assets/screenshots/agent-watch.png" alt="Watching the harness live" width="420"><br>**Watch the harness live** (`w` during a run): one screen, no scrolling: the earlier steps one line each, the current one in full (`e`: whole transcript) | <img src="assets/screenshots/agent-tasks.png" alt="The same task with three harnesses" width="420"><br>**The same task, three harnesses**: resolved or not, steps and how each one ended |
| <img src="assets/screenshots/agent-report.png" alt="Agent run report" width="420"><br>**Report**: resolved per harness, patch applied, steps, FAIL_TO_PASS | <img src="assets/screenshots/agent-export.png" alt="Export as a dataset" width="420"><br>**Export** (`x` in Results): JSONL or Parquet |

## The TUI

<p align="center">
  <img src="assets/screenshots/main.png" alt="slmbench home screen: the model to evaluate with its estimated memory, and the menu" width="720">
</p>

<p align="center"><img src="assets/screenshots/splash.png" alt="The animated loading screen: the slmbench wordmark, the benchmark bars and the start-up checks" width="520"><br><sub>The loading screen at start: it connects to the model server, checks the sandboxes and loads the leaderboard (any key skips it).</sub></p>

The home screen always shows **the model you're evaluating**: its parameters, quantization, server, leaderboard row, **estimated memory** and connection. The bottom-right corner shows any evaluation in progress, even one started from another terminal; click it or press `Ctrl+R` to watch it.

| | |
|:-:|:-:|
| <img src="assets/screenshots/run.png" alt="Run benchmarks" width="420"><br>**2 · Run benchmarks**: the mode, its benchmarks and the options | <img src="assets/screenshots/live.png" alt="Live view of a running evaluation" width="420"><br>**Live view**: progress, ETA, tokens/s, per-benchmark bars and the latest answers |
| <img src="assets/screenshots/report.png" alt="Run report" width="420"><br>**4 · Reports** with pass@1 / pass@5, and side-by-side comparisons | <img src="assets/screenshots/review.png" alt="Failure explorer" width="420"><br>**Failure explorer** (`e`): each answer and why it failed |
| <img src="assets/screenshots/leaderboard.png" alt="Leaderboard" width="420"><br>**5 · Leaderboard** per model and quantization | <img src="assets/screenshots/card.png" alt="Benchmark card" width="420"><br>**3 · Benchmark cards** with the exact prompt the model receives |

<details>
<summary><b>Keys and screens</b></summary>

- **1 · Model to evaluate**: a local `.gguf` (slmbench starts llama-server for you) or a running server; API key; connection test.
- **2 · Run benchmarks**: mode and benchmarks (`?` shows a benchmark's card), samples, max tokens and the **reasoning budget**. While it runs, **`m` goes back to the main menu** and the run keeps going in the background (`q`/`Esc` offer that, cancelling or staying); the menu tells you when it ends and opens its report, and quitting the TUI with runs going asks first (they're cancelled, finished results kept).
- **Live view** (`Ctrl+R`, or click "● running"): progress and ETA, per-benchmark progress, tokens/s, empty or truncated answers, the latest answers and lighteval's log (`l`). `←→` switches between runs, `x` cancels one. In an agent run, **`w` watches the harness live**, laid out to fit one screen: the earlier steps one line each (what they ran and how it went) and the current step in full (the end of its reasoning, the answer, the command or tool call and its output), updated as it happens; `e` switches to the whole transcript, `t` picks another task. `w` works on the run screen too.
- **4 · Results**: `Enter` report, **`e` explore the answers** (benchmark → problems, failures first → prompt, answers and why each one failed; code is re-checked in the sandbox), **`l` errors and log**, **`r` rescore**, **`u` resume**, `Space` + `c` compare.
- **7 · Chat with the model**: try a local GGUF by hand through `llama-cli`; `/exit` comes back.
- **8 · Model memory**: see [below](#will-it-fit-memory-estimates-with-hf-mem).
- **9 · Agent traces**: the agent-mode runs → their tasks (resolved or not, steps, cut answers) → every step: reasoning, answer, bash command or tool call with its arguments, and its output; `n`/`p` move between steps. Also `e` on an agent run in Results.
- **4 · Results → `x`**: export the marked runs, this one, every agent run or every run as a dataset (JSONL or Parquet).
- `L` switches between **English** and **Spanish**.
- At start, an **animated loading screen** while slmbench connects to the model server, checks the sandboxes and loads the leaderboard (any key skips it; *6 · Settings* or `SLMBENCH_NO_SPLASH=1` turn it off).
- Works on macOS, Linux and **WSL** (Windows Terminal). It needs a UTF-8 locale for its borders and wordmark: if `LANG` is unset (common on WSL) it switches to `C.UTF-8` by itself, and with no UTF-8 locale at all it falls back to plain ASCII. The large wordmark shows from 80×37, a smaller one from 42×31; smaller terminals just skip it.

</details>

## Will it fit? Memory estimates with hf-mem

Before you download a model or start a long run, slmbench tells you **how much memory it needs and whether it fits on this machine**, with [hf-mem](https://github.com/alvarobartt/hf-mem): it reads the Safetensors / GGUF metadata (HTTP range requests, or the header of a local file) **without downloading the weights**, and adds up the weights, the **KV cache for your context** and the runtime overhead.

| | |
|:-:|:-:|
| <img src="assets/screenshots/memory.png" alt="Memory of LFM3-1.7B Q8_0: 2.7 GB, fits, with the weights, KV cache and overhead" width="420"><br>**8 · Model memory**: the total and whether it fits, what it's made of, and the context (`←→`) | <img src="assets/screenshots/memory-quants.png" alt="Every GGUF quant of LFM3-1.7B with its memory" width="420"><br>**A whole GGUF repo**: every quant side by side; `Enter` opens one |

- **On the home screen** the *memory* line shows it for the model you're evaluating, and **8 · Model memory** opens the full view. `←→` changes the context (the KV cache grows with it), `m` estimates any other model (`org/repo`, `org/repo:Q4_K_M` or a path to a `.gguf`) and `t` shows it as a table.
- **Which file?** The GGUF that llama-server is serving (when you started it from slmbench), a `.gguf` path, or a Hugging Face id. A plain id like `LiquidAI/LFM3-1.7B` is looked up as its GGUF repo first (`LiquidAI/LFM3-1.7B-GGUF`, in the quant the server reports, or Q4_K_M), since GGUF is what you serve with llama.cpp; then the repo's quants; then the original safetensors.
- **1 · Model to evaluate → A · Local model** shows the estimate live as you pick the `.gguf` and change the context (`-c`).
- From the command line:

```bash
slmbench mem LiquidAI/LFM3-1.7B-GGUF:Q4_K_M --ctx 32768     # one quant, for a 32k context
slmbench mem LiquidAI/LFM3-1.7B-GGUF                         # every quant of the repo
slmbench mem ~/models/model.gguf --json                      # a local file, as JSON
```

## Choosing the model to evaluate

slmbench doesn't load models itself: it talks to a **server with an OpenAI-compatible API**. The *model to evaluate* is the pair *server + model id*, shown on the home screen and saved in `~/.config/slmbench/config.json`.

| | How |
|---|---|
| **Local GGUF** | TUI → *1 · Model to evaluate* → **A · Local model**: pick a `.gguf` (found in the Hugging Face cache, LM Studio…) or an `org/model:quant` repo and press *Start server* |
| **Running server** | TUI → *1 · Model to evaluate* → **B · Connect**: llama-server `:8080`, Ollama `:11434`, LM Studio `:1234`, vLLM `:8000`, **Unsloth Studio** `:8888` or any URL |
| **Unsloth Studio** | **B · Connect** → *Unsloth Studio* (`http://127.0.0.1:8888/v1`). Studio always asks for an API key: slmbench takes it from `UNSLOTH_API_KEY` when it's set (only for Studio, never for another server, and without saving it), otherwise it asks for it and keeps it like any other key |
| **CLI** | `slmbench models` and `slmbench use <model> --base-url <url>`; or just for one run: `slmbench run --model … --base-url …` |
| **Hosted API** (OpenRouter, Together, Groq, OpenAI…) | **B · Connect** with the provider's URL and fill in *API key*; or `slmbench run --base-url https://openrouter.ai/api/v1 --api-key "$KEY" --model …` |
| **Local GGUF + LoRA adapter** | **A · Local model** → *LoRA adapter*: a `.gguf` adapter, a Hugging Face repo (`org/repo[:file.gguf]`) or its URL, plus an optional *scale*. See below |

Each leaderboard row is **model + quantization** (taken from llama-server or from the model id), so the same model in Q4_K_M and Q8_0 are compared side by side.

### LoRA adapters

Fine-tunes published as LoRA adapters, such as [harness-hub/LFM3-1.7B-opencode](https://huggingface.co/harness-hub/LFM3-1.7B-opencode) (LFM3-1.7B trained for the OpenCode harness), are measured on top of their base model: pick the base GGUF in **A · Local model** and give the adapter in *LoRA adapter*. slmbench downloads it to the Hugging Face cache and starts llama-server with `--lora` (`--lora-scaled` if the scale isn't 1), the same as:

```bash
hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf --local-dir .
hf download harness-hub/LFM3-1.7B-opencode-GGUF LFM3-1.7B-harbor-opencode-lora-F16.gguf --local-dir .
llama-server -m LFM3-1.7B-Q8_0.gguf --lora LFM3-1.7B-harbor-opencode-lora-F16.gguf --alias harness-hub/LFM3-1.7B-opencode
```

- **What you can give**: a local `.gguf` adapter; a repo (`harness-hub/LFM3-1.7B-opencode-GGUF`: the adapter file is found in it, the F16 one if there are several) or `org/repo:file.gguf`; or the Hub URL of the repo or the file.
- **PEFT adapters** (`adapter_model.safetensors`, which llama.cpp can't load): slmbench uses the repo's `-GGUF` version when it exists, so the URL of `harness-hub/LFM3-1.7B-opencode` works as is. Otherwise, convert it with llama.cpp's `convert_lora_to_gguf.py` and give the resulting file.
- **Its own row and a ranking**: the model's default alias becomes the adapter's name (`harness-hub/LFM3-1.7B-opencode`), so base and fine-tune are separate leaderboard rows; the adapter loaded (read from llama-server's `/lora-adapters`) and its base model are recorded, and the leaderboard's *Adapters* ranking shows how much each adapter adds over its base.
- **Unsloth Studio**: connect to it (**B · Connect** → *Unsloth Studio*) and use the *Unsloth Studio* section of *1 · Model to evaluate*: pick the base GGUF among the ones Studio has, give the *LoRA adapter* (same forms as above) and the *Context* (tokens; 0 lets Studio fit it to the memory, and it may grant less than asked, which the load message says) and press *Load in Unsloth Studio*. slmbench downloads the adapter and asks Studio to load the model with it (`/v1/load` with `--lora` for its llama-server), checks that Studio applied it, and makes it the model to evaluate. Runs go to the adapter's leaderboard row (Studio names the model after its base) with the base recorded for the ranking; loading the base without an adapter really drops it (Studio would otherwise keep the previous load's). Loading the PEFT adapter itself (`is_lora`) goes through Studio's MLX backend, which can't apply LFM3 adapters yet.
- From the CLI: `slmbench config set srv_lora harness-hub/LFM3-1.7B-opencode-GGUF` (and `srv_lora_scale`). With a server you run yourself, start it with `--lora` and connect to it (**B**): the adapter is detected and recorded the same way.

<details>
<summary><b>Hosted APIs</b>: what is sent and what to keep in mind</summary>

With a hosted API, slmbench sends only standard OpenAI parameters (the reasoning budget is a llama.cpp parameter and isn't sent). If the provider still rejects one, for example `max_tokens` or `temperature` on OpenAI's reasoning models, slmbench adapts it once for the whole run (`max_completion_tokens`, default temperature…) and notes the change in the report. Keep in mind that pass@k multiplies the requests of coding and tool calling by 5 (start with `--max-samples`), that prompts are sent to the provider, and that its speed isn't comparable with your machine.

</details>

## Leaderboard

Every run is added automatically to **[leaderboard.md](leaderboard.md)** (with an SVG chart) and to `leaderboard/leaderboard.html`: an interactive page with one tab per mode, a score-by-model chart, a score-against-speed chart and a sortable table, in light and dark mode.

- Results are stored **per benchmark** in the model + quant row: run everything, one mode or a single benchmark, and the row fills up as you go (a later run of the same benchmark replaces the earlier one).
- A mode's **Score** is the mean of its benchmarks. Only rows with **every benchmark of the mode measured on the full set** are ranked first and get medals; the others are listed below, marked `⚠️ k/N` (k of N benchmarks measured) or `partial` / `n=` (run with `--max-samples`, indicative only).
- A partial run **never replaces** a full result already in the leaderboard.
- **tok/s** is measured per mode: long-context prompts are much slower, so that table shows its own speeds.
- **LoRA adapters**: models served with a LoRA adapter get their own ranking (in leaderboard.md, an *Adapters* tab in the HTML page and in the TUI): each adapter against its base model, with the **gain in points** per mode, measured on the benchmarks both rows share, ranked by the mean gain. The base is the GGUF the server was started with (or the adapter's `base_model` on Hugging Face): measure it too, same quant, to see the gain.
- **Harnesses**: the agent mode also gets its own section (in leaderboard.md, a tab in the HTML page and in the TUI): a card per harness (version, how the model acts, mean resolved over the models measured with it, its best model) and the models × harnesses table with each model's best harness highlighted. The same model can resolve very different numbers of tasks depending on the agent that drives it.

## Cheat sheet

```bash
slmbench                                   # TUI
slmbench setup --mode general|code|tools|long|all   # prepare a mode
slmbench run [--mode code] [--benchmarks a,b] [--max-samples N] [--max-tokens N] [--reasoning-budget N]
slmbench run --resume <run>                # continue an interrupted run (reuses its answers)
slmbench rescore <run>                     # score a run again from its saved answers, without the model
slmbench review <run> [--task KEY] [--sample N] [--all]   # what the model answered and why it failed
slmbench log <run>                         # a run's errors: failed requests grouped by message, warnings, tracebacks
slmbench benchmarks [key]                  # catalog, or one benchmark's card with a sample task
slmbench models | use <model> --base-url URL
slmbench mem <org/repo[:quant] | model.gguf> [--ctx N] [--json]   # memory it needs (hf-mem), without downloading
slmbench chat [model.gguf | org/repo:quant] [--system TEXT]   # chat with a local GGUF via llama cli
slmbench results | compare <dir> <dir>
slmbench leaderboard [open]
slmbench config set ui_lang es             # TUI in Spanish
```

## Browsing pi.dev (Pi + Chrome DevTools MCP)

`slmbench browse` lets you **watch the model use a web browser**: it drives a real Chrome through the
[Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp) server with the [Pi](https://pi.dev) agent as
its harness, on the **pi.dev** website. Chrome opens in a window and Pi's interface shows every step in the terminal.

![slmbench browse: LFM3-1.7B clicks, scrolls and filters on pi.dev](assets/browse-demo.gif)

*Real sessions (2× speed, [MP4](assets/browse-demo.mp4)) with LFM3-1.7B (Q4_K_M), one task each: **clicks** (open the
newest release in the Changelog), **scrolling** (PageDown to the section on Pi's modes) and **filters** (type a name, pick a
package type and press SEARCH on the Packages page).*

```bash
slmbench browse                                   # demo: type in Pi what it should do on pi.dev (Ctrl+D to leave)
slmbench browse "Open the page that lists the models"     # demo with a first task
slmbench browse --tasks my-tasks.json             # your tasks, each checked and scored; report in the results folder
```

It is not one of the benchmark modes: the tasks are yours (slmbench ships none), and the scored runs are not added to the
leaderboard. Task file format, checks and requirements (Node.js ≥ 22.19; Pi, Chrome DevTools MCP and Chrome for Testing are
installed on first use) in **[docs/browse.md](docs/browse.md)**.

## Built on lighteval

lighteval provides **the tasks and the metrics**; slmbench provides **the model client, the interface and the results**:

- Benchmarks that ship with lighteval (IFEval, GSM8K, MATH-500) are used as they are.
- The ones it lacks (IFStruct, MGSM-es, BFCL, RULER, HumanEval+, MBPP+, CRUXEval, MultiPL-E) and LiveCodeBench with pass@k are implemented as lighteval tasks in `bench/slmbench/benchmarks/`, reproducing the official prompt and metric. To add another one, see **[docs/adapting-a-benchmark.md](docs/adapting-a-benchmark.md)**.
- The model is a small custom client (`lighteval custom`) that talks to your server over the OpenAI API (including native tool calling), without litellm. It saves every answer with the run, records speed, empty and truncated answers, and reports API errors.

## Tips

- **Start small**: a full benchmark with a reasoning SLM can take hours, so try `--max-samples 20` first.
- **pass@5 costs 5×**: coding and tool calling make 5 requests per problem; `--passk-n 1` gives a quick pass@1-only run.
- **Answer length**: by default each answer gets at most **2048 tokens**, of which up to **1024** can be reasoning (`--max-tokens`, `--reasoning-budget`; `-1` = unlimited). If a reasoning model leaves empty or truncated answers (they score 0), raise `--max-tokens` or lower `--reasoning-budget`.
- **LiveCodeBench** problems are long: answers get at least 8,192 tokens, and small reasoning models do better with `--max-tokens 16384`. slmbench ships a fix for lighteval's LiveCodeBench checker, which scores every answer 0 on macOS.
- **Coding benchmarks need the sandbox**: without it, slmbench refuses to run them (see [Security](#security)).

## Security

The coding benchmarks run code written by the model you're testing. **slmbench never runs it on your machine**: it runs in a [Docker Sandboxes](https://docs.docker.com/ai/sandboxes/) microVM with its own kernel, **no network, no access to your files**, as an unprivileged user, with limits and a timeout. Your API key is only sent to the server you configured and never saved with the runs.

<details>
<summary><b>The details</b>: the sandbox, browsing and API keys</summary>

> **About `dhi.io` and other domains in `sbx policy ls`:** Docker Sandboxes ships a global default policy that allows about 200 domains (package registries, AI APIs, `dhi.io`: the Docker Hardened Images registry the sandbox images come from) and allows host-path access for sandboxes that mount a workspace. slmbench doesn't add those rules and they don't reach the `slmbench-run-…` sandboxes: each one is created with its own deny-all network rule and no workspace.

The coding benchmarks run code written by the model you're testing. slmbench never runs it on your machine:

- **Where:** a [Docker Sandboxes](https://docs.docker.com/ai/sandboxes/) microVM with its own Linux kernel (not a container sharing your kernel), half of your CPU cores and 2 GB of memory (the checks peak at about 0.7 GB).
- **One per run, deleted at the end:** `setup` builds the sandbox once and saves it as a local template (`slmbench-exec:v2`). Each run creates a new `slmbench-run-…` sandbox from it in a few seconds and deletes it when the run ends, whether it finished, failed or was cancelled. Re-checking answers (`slmbench review`, `e` in the TUI) creates one the same way and deletes it after 2 minutes without checks or when slmbench exits. A sandbox left behind by a process that died is deleted the next time one is created.
- **No access to your files:** it's created without a host workspace (only `/etc/resolv.conf` is shared, read-only). Each check gets a fresh temporary directory inside the sandbox, deleted afterwards.
- **No network:** each sandbox is created with a deny-all rule, so it never has network. During `setup` only the Debian mirror is allowed, while the compilers are installed, and that rule is removed right after.
- **Unprivileged inside the sandbox too:** the sandbox's default user has `sudo`, so model code never runs as it. Each check runs as one of the users `slmrun0…7`: no sudo, no groups, no home, `no_new_privs` (setuid binaries can't raise it), in a directory only that user can open. It can't write to the system, the tools or the scoring harness, nor read other checks. Limits: 128 MB per file, 512 processes (no fork bombs). When the check ends, every process of that user is killed, also anything left in the background, and its temporary files are deleted. The persistent `slmbench-exec` of an older slmbench is turned into the template by `slmbench setup --mode code` (no reinstall) and removed.
- **Clean environment:** the code runs with only `PATH`, `HOME`, `LANG` and the Go settings it needs, plus a timeout per step.
- **Fast:** a persistent worker inside the sandbox receives the checks over one `sbx exec` and runs several at a time (the n answers of a pass@k problem in parallel): 3–6× faster than one `sbx exec` per check.
- **No fallback:** if the sandbox isn't ready, coding benchmarks don't start. lighteval's own LiveCodeBench tasks (`lcb:*`), which run code on the host, are refused; slmbench's `lcb` is the same benchmark run in the sandbox.

Checked from inside the sandbox: model code runs as `slmrun0` with `sudo` refused and writes to `/usr`, `/etc`, the harness and other checks' folders denied; a fork bomb is contained and cleaned up; your home folder and files aren't visible, files written there don't appear on the host, HTTP requests are rejected, raw TCP connections deliver no data, UDP doesn't get out, and the only credential-looking variable the sandbox has is a placeholder (not your token), which the code doesn't even see.

### Agent mode

The agent's commands run in a container with no network inside a sandbox that has **no network either**: the model is
reached through a relay over `sbx exec` that only lets chat completions through, to the model under test. The task images
are fetched by a separate sandbox and piped in. Your API key never enters the sandbox. Details in
[docs/agent.md](docs/agent.md#where-it-runs).

### Browsing pi.dev

`slmbench browse` gives the model a real browser, so it is limited to one website: only `pi.dev` and its subdomains resolve
in that Chrome (any other link doesn't load). The model gets only the Chrome DevTools MCP browser tools (snapshot, click,
fill, type, key, hover, navigate): no shell or file tools, and JavaScript evaluation is off. Chrome uses a throwaway profile
and never the system keyring, and Pi uses its own configuration folder (your `~/.pi` is not read).

### API keys

- The key is stored only in `~/.config/slmbench/config.json` (permissions 600) and sent only to the configured server: never to Hugging Face, the leaderboard or the sandbox, and dropped if the server redirects to another host.
- Runs never save it: `run.json` leaves it out, `model.json` shows `***`, and error messages from the provider are scrubbed before they reach `requests.jsonl`, the log or the report, even when the provider echoes the key back.
- The `sbx` processes are started without any secret-looking environment variable.
- **Contributing:** run `git config core.hooksPath .githooks` once per clone. The `pre-commit` hook blocks any commit whose staged files contain the API key configured in slmbench, a token from the environment, or anything shaped like a key (OpenAI/OpenRouter, Hugging Face, GitHub, AWS, Google, Slack, private keys, `"api_key": "…"`), binaries included, without printing the value.

</details>

## Documentation

| | |
|---|---|
| [docs/benchmarks.md](docs/benchmarks.md) | **Benchmarks used**: what each one measures, dataset, metric, origin, validation and a real sample task |
| [docs/adapting-a-benchmark.md](docs/adapting-a-benchmark.md) | How to adapt a Hugging Face benchmark so slmbench can run it |
| [docs/lighteval.md](docs/lighteval.md) | How lighteval is integrated (custom model, no litellm) and version pitfalls |
| [docs/llama-cpp.md](docs/llama-cpp.md) | Serving GGUF models with llama.cpp: getting models and an argument reference |
| [docs/agent.md](docs/agent.md) | **Agent mode**: SWE-bench Lite with mini-SWE-agent, Pi or OpenCode, the no-network sandbox, arm64 images, trajectories |
| [docs/browse.md](docs/browse.md) | `slmbench browse`: the model browses pi.dev with Pi + Chrome DevTools MCP (demo, or your tasks scored) |
| [docs/omp.md](docs/omp.md) | Using local models as a coding agent with *oh my pi* |

## License

slmbench is released under the [MIT License](LICENSE), like lighteval. The IFStruct validator in `bench/slmbench/benchmarks/ifstruct_validator.py` is vendored unchanged from [Liquid4All/ifstruct](https://github.com/Liquid4All/ifstruct) under the [Apache License 2.0](licenses/ifstruct-LICENSE). Each benchmark dataset keeps its own license.

<p align="center">
  <img src="assets/logo.jpg" alt="slmbench logo" width="360">
</p>
