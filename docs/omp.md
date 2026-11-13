# oh my pi (`omp`): a guide to using it with llama.cpp

`omp` v18.x. This guide covers setting it up with a local llama.cpp server, its modes, permissions and the recommended configuration for small models. To start the server, see [llama-cpp.md](llama-cpp.md).

## 1. Oh my pi (`omp`) with llama.cpp

`omp` is a coding agent for the terminal (like Claude Code). You run it from the project directory.

### 1.1 Setup

omp reads providers from `~/.omp/agent/models.yml` and settings from `~/.omp/agent/config.yml`. Back them up before editing them (`cp models.yml models.yml.bak`).

**llama.cpp provider** in `~/.omp/agent/models.yml` (the `id` must match `llama-server`'s `--alias`, and `contextWindow` its `-c`):

```yaml
providers:
    llamacpp:
        api: openai-completions
        auth: none
        baseUrl: http://127.0.0.1:8080/v1
        models:
            - id: lfm3-1.7b
              name: LFM3-1.7B (llama.cpp)
              reasoning: true
              input:
                - text
              contextWindow: 32768
              maxTokens: 8192
```

**Default model** (saved in `config.yml` as `modelRoles.default`):

```bash
omp config set modelRoles.default llamacpp/lfm3-1.7b
```

If the default model is a llama.cpp one, **`llama-server` must be running on the `baseUrl` port** before you launch `omp`.

**Disabling a provider** you don't want to see (e.g. a cloud provider whose key is in your environment for other tools):

```bash
omp config set disabledProviders '["openrouter"]'   # disable
omp config set disabledProviders '[]'               # re-enable them all
```

### 1.2 Workflow

```bash
# 1. Terminal A: server (see llama-cpp.md, sections 1 and 4; $MODEL points to your .gguf)
llama-server -m "$MODEL" --alias lfm3-1.7b --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 -fa on --jinja
#    or, without downloading first:  llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q4_K_M --alias lfm3-1.7b --port 8080 -c 32768 --jinja

# 2. Terminal B: in the project directory
cd ~/my-project
omp
```

### 1.3 Using omp

```bash
omp                                      # interactive
omp "List the .ts files in src/"         # interactive with a first message
omp -p "Summarize @README.md"            # one answer, then exits (tested)
omp @doc.pdf @img.png "What's in it?"    # attach files with @
omp -c                                   # continue the last session
omp -r                                   # pick a previous session
omp --model llamacpp/lfm3-1.7b           # specific model (provider/id)
omp --model qwen                         # fuzzy name search
omp --models llamacpp/lfm3-1.7b,ollama/<model>   # models to cycle through with Ctrl+P
omp --thinking off                       # no reasoning (faster with small models)
omp --no-session                         # don't save the session
omp --max-cost 1                         # spending cap in USD
```

Useful commands:

| Command | What for |
|---|---|
| `omp models` | List and search models |
| `omp config list` / `get` / `set <key> <value>` | View and change settings |
| `omp stats` | Usage statistics |
| `omp commit` | Generates a commit message |
| `omp update` | Updates omp |

In interactive mode: `Ctrl+P` cycles through the `--models` models; `/` commands (`/login`, `/share`, …) open menus; the model picker (`/model`) also lists the models defined in `models.yml`, with their `name`.

### 1.4 Seeing the models available in omp

`omp models` lists every model from the active providers. Typical local setups:

| Provider/model | Requirement |
|---|---|
| `llamacpp/<alias>` | `llama-server` running on the `baseUrl` port |
| `ollama/<model>` | Ollama on port 11434 with the model pulled (`ollama list`) |
| Cloud providers | Their API key in the environment, or `/login` inside omp |

### 1.5 Adding another llama.cpp model to omp

1. Start the server with `--alias NAME` (and a different `--port` if one is already running).
2. In `models.yml`, the `id` must match the alias. Same port: another entry under `models:` of the `llamacpp` provider. Different port: another provider with its own `baseUrl`. Example (a vision model served with `--alias my-vision-model --port 8081`):

```yaml
    llamacpp-vision:
        api: openai-completions
        auth: none
        baseUrl: http://127.0.0.1:8081/v1
        models:
            - id: my-vision-model
              name: Vision model (llama.cpp)
              input:
                - text
                - image
              contextWindow: 32768
              maxTokens: 8192
```

3. Use it with `omp --model llamacpp-vision/my-vision-model`.

### 1.6 Known limitations

- A ~2B model such as LFM3-1.7B may fail with complex tools. For serious agent work, use a larger model (e.g. 7-14B).
- If a configured MCP server fails to connect at startup (e.g. "closed stdout before responding"), its tools won't be available; check `~/.omp/agent/mcp.json` or `/mcp`.

### 1.7 Changing omp's system prompt

```bash
omp --append-system-prompt "Always answer in Spanish."       # appends to the default prompt (recommended)
omp --append-system-prompt @my-rules.md                      # from a file
omp --system-prompt "You are a pirate."                      # replaces it entirely
omp --system-prompt-template template.hbs                    # Handlebars template (can't be combined with --system-prompt)
```

Without flags, omp loads files automatically (walking up from the working directory, and from your home directory):
- `AGENTS.md` / `CLAUDE.md`: persistent project instructions (the recommended option).
- `SYSTEM.md` / `SYSTEM_TEMPLATE.md`: a custom system prompt, in `.omp/`, `.agent/`, `.agents/` or `.claude/SYSTEM.md`.
- `APPEND_SYSTEM.md`: text permanently appended to the prompt.
- `omp config set contextFiles.extra '["AGENTS.local.md"]'`: additional context file names.
- `--no-context-files` disables `AGENTS.md`/`CLAUDE.md`. Changes take effect in a new session.

With a 1.7B model, prefer a short `--append-system-prompt` or `AGENTS.md`: replacing the whole prompt removes the tool-use instructions, and a long `AGENTS.md` eats into the context (with `-c 32768`, just 32k).

## 2. omp modes (version 18.x)

List taken from omp's own help (`omp --help`, `omp config list`) and from the strings inside the binary. Inside omp, type `/` to see every command available in your version.

### 2.1 Working modes (enabled with `/` commands inside omp)

| Mode | Command | What it does |
|---|---|---|
| **Plan** | `/plan` (or `omp --plan-yolo`) | The agent plans in read-only mode before executing. When it's done, the plan is reviewed and approved. `/plan-review` reopens the review of the last plan. Settings: `plan.enabled`, `plan.defaultOnStartup`, `plan.autosave` |
| **Vibe** | `/vibe` | You direct persistent worker sessions (fast/good). The main agent only reads, and launches, messages and waits for the workers. Can't be combined with plan or goal |
| **Goal** | `/goal <goal>` or `omp --goal "..."` | A persistent autonomous goal for the session. Subcommands: show details, pause, resume, drop and adjust the token budget (`<N\|off>`). There's a command to have the agent interview you before setting the goal. Settings: `goal.enabled`, `goal.continuationModes` |
| **Loop** | `/loop [N\|duration] [--while\|--until '<cmd>'] [prompt]` | Repeats the prompt after every turn. E.g.: `/loop 10`, `/loop 10m`, `/loop 20 --until 'bun test' fix the tests`. Another `/loop` turns it off. Settings: `loop.mode` = `prompt`, `compact` or `reset` |
| **Fast** | `/fast [on\|ultra\|off\|status]` | Priority service (`ultra` = ultrafast). Only on cloud models that offer it; doesn't apply to llama.cpp |
| **Slow** | `/slow [on\|off\|status]` | Flex tier (cheaper, higher latency) on OpenAI/Google; on Anthropic, low priority once the session limit is reached. Doesn't apply to llama.cpp. Not to be confused with the `slow` model role (2.2) |
| **Extended context** | `/extended-context` | Larger context windows on models that offer them |
| **Advisor** | `/advisor` or `omp --advisor` | A second model that reviews every turn and injects notes. Settings: `advisor.enabled`, `advisor.reviewMode` (`turn`/`agent-end`) |
| **Prewalk** | `/prewalk` or `omp --prewalk` | Switches to a fast/cheap model at the first edit or write, once the plan is done. Settings: `prewalk.enabled`, `prewalk.into` (default `@smol`) |
| **Plan-yolo** | `omp --plan-yolo [--plan-yolo-into <model>]` | Read-only plan, automatic approval and execution with another model (default `smol`) |
| **Live (voice)** | `/live` | Live voice session with the agent |
| **Computer use** | its own command | Enables the computer-use tool in the session |
| **Ratchet** | `/ratchet [workflow and goal]` | Builds an evaluation for an LLM workflow and optimizes it unattended |
| **Vim** | `tui.vimMode` setting | Edit the prompt with Vim keys |
| **Collab** | `/collab`, `omp collab`, `omp join` | Share the session live, read-only or with control |

Other session commands: `/model` (switch model), `/modelpreset` (save and switch presets of roles and reasoning level), `/mcp` (manage MCP servers), `/marketplace` and `/plugins`, `/session`, `/usage`, `/skills` (list of skills), `/changelog`.

### 2.2 Model roles

omp assigns a model to each role. If you only define `default`, the other roles (`smol`, `slow`, `plan`) fall back to it.

| Role | Use | How to set it |
|---|---|---|
| `default` | Normal conversation | `modelRoles.default` or `--model` |
| `smol` | Light, quick tasks | `--smol <model>`, env `PI_SMOL_MODEL` |
| `slow` | Deeper analysis | `--slow <model>`, env `PI_SLOW_MODEL` |
| `plan` | Architecture planning | `--plan <model>`, env `PI_PLAN_MODEL` |

`cycleOrder` (default `smol`, `default`, `slow`) sets the order when cycling models. Example with local models:

```bash
omp --model llamacpp/lfm3-1.7b --smol llamacpp/lfm3-1.7b --slow ollama/<large-model> --plan ollama/<large-model>
omp config set modelRoles.slow ollama/<large-model>     # set it permanently
```

### 2.3 Reasoning level

`omp --thinking <level>` with `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or `auto` (default `auto`). Setting: `defaultThinkingLevel`. With small reasoning models, `--thinking off` reduces latency.

### 2.4 Tool permissions (`tools.approvalMode`)

| Value | Behavior |
|---|---|
| `always-ask` | Asks for confirmation on every tool |
| `always-ask-cached` | Asks and remembers the answers |
| `write` | Asks only before writes |
| `yolo` | Approves everything without asking |

`yolo` is omp's default; with small local models, `write` or `always-ask` is a better idea (see section 3). To change it:

```bash
omp config set tools.approvalMode write      # only ask before writing/editing
omp --approval-mode always-ask               # for this session only
omp --auto-approve                           # approve everything in this session
```

### 2.5 Output modes (`--mode`)

| Value | Use |
|---|---|
| `text` | Plain text (default) |
| `json` | JSON output, for scripts |
| `rpc` | RPC protocol to integrate it into other tools |
| `rpc-ui` | RPC with a tool interface |
| `omp acp` | ACP (Agent Client Protocol) server over stdio, for editors |

`-p` is the non-interactive mode; `--print-thoughts` includes the reasoning in that output.

### 2.6 Edit modes (`edit.mode`)

`hashline` (default), `apply_patch`, `patch`, `replace`, `sloppy`, `none`. To change it: `omp config set edit.mode patch`.

### 2.7 Other behavior settings

- `steeringMode` and `followUpMode`: `one-at-a-time` or `all`, for the messages you send while the agent is working.
- `interruptMode`: `immediate` or `wait`.
- `autoResume`: resume sessions automatically.
- `task.*`: subagents (`task.maxConcurrency`, `task.isolation.enabled`, `task.autoParallel`, …).
- `--profile <name>`: an isolated profile for credentials, sessions and settings; `--alias` creates the shell shortcut.
- `--max-time`, `--max-cost`, `--warn-cost`: time and spending limits.

## 3. Recommended omp configuration (small local model)

All these commands are accepted by omp 18.x. They are saved in `~/.omp/agent/config.yml`. Back it up first: `cp ~/.omp/agent/config.yml ~/.omp/agent/config.yml.bak`.

```bash
# --- Safety (a 1.7B model makes more mistakes with tools) ---
omp config set tools.approvalMode write        # ask before writing/editing (always-ask: always ask)
omp config set bash.gitGuard true              # blocks destructive git commands
omp config set tools.restrictedFolders '["~/.ssh","~/.omp","~/.aws","~/.gnupg"]'   # off-limits folders

# --- Providers and model ---
omp config set disabledProviders '["openrouter"]'
omp config set modelRoles.default llamacpp/lfm3-1.7b
omp config set defaultThinkingLevel low        # less reasoning = faster answers with LFM3

# --- 32k context (the server's -c) ---
omp config set compaction.thresholdPercent 70  # compact when the context is 70% full
omp config set compaction.keepRecentTokens 8000   # default 20000, too much for 32k

# --- Performance with a single local server ---
omp config set task.maxConcurrency 2           # parallel subagents (default 32)

# --- Startup ---
omp config set startup.checkUpdate false
omp config set update.auto false               # update by hand with `omp update`
```

Why:
- **Safety:** `write` or `always-ask` keeps a small model from deleting or overwriting files without you noticing.
- **Concurrency:** `llama-server` splits `-c` among its slots (`-np`). With 32 subagents and a single slot, everything gets queued. If you use 2, start the server like this: `llama-server ... -np 2 -c 65536` (32k per slot).
- **Compaction:** with a 32k context, omp's default threshold (designed for huge windows) may run out of room before compacting.
- **A separate profile:** `omp --profile local --alias omp-local` creates a shortcut with isolated credentials, sessions and settings, so local use doesn't get mixed with other providers.

View and revert:

```bash
omp config list --scope global      # what you have set yourself
omp config reset <key>              # back to the default value
```

Optional settings depending on the case:

```bash
omp --thinking off -p "..."                  # a quick query without reasoning
omp --max-cost 1 --max-time 30m              # limits for long tasks with cloud models
omp config set edit.mode patch               # if hashline fails with a small model
```

## 4. Tips

- A 1.7B model may fail with complex tools in omp; for serious agent work, use a larger model (e.g. 7-14B), local or in the cloud.
- If the default model is a llama.cpp one, `llama-server` must be running on the port configured in `models.yml` before you launch `omp`.
- If omp warns that an MCP server can't connect, it doesn't affect using it with llama.cpp; check `~/.omp/agent/mcp.json`.
- Updating: `omp update`.
