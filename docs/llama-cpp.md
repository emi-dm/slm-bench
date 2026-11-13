# llama.cpp: a guide to running local models

A reference guide for serving GGUF models locally with llama.cpp (`llama-cli`, `llama-server`). Tested on Apple Silicon with llama.cpp b12110 installed with Homebrew (`brew install llama.cpp`); the arguments are the same on Linux and Windows.

> **Shortcut:** if you only want to evaluate a model with this repo, you don't need to type these commands by hand. The slmbench TUI (`uv run slmbench` → "1 · Model to evaluate" → "A · Local model") finds the GGUF files on your disk and starts and stops `llama-server` for you.

## 1. Where to find or how to get GGUF models

Only `.gguf` files work with llama.cpp. MLX or `safetensors` (Transformers) models **don't**; for those, look for the GGUF version of the same model on Hugging Face (usually named `<model>-GGUF`).

Common locations of already downloaded GGUF files:

| Source | Typical path |
|---|---|
| Hugging Face cache (`hf download`, `huggingface_hub`) | `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/*/` |
| llama.cpp cache (`-hf`) | `~/.cache/llama.cpp/` (on macOS it may be `~/Library/Caches/llama.cpp/`); list it with `llama-server --cache-list` |
| LM Studio | `~/.lmstudio/models/<org>/<repo>/` |

To find every GGUF on disk: `find ~ -name '*.gguf' -size +10M 2>/dev/null`. `mmproj-*.gguf` files are not models but vision or audio projectors (`--mmproj`) that go with the main model. Ignore partial downloads (`*.part`, `*.incomplete`).

Downloading a model:

```bash
# With llama.cpp: downloads to its cache (plus the mmproj, if any) and serves it
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q4_K_M

# With the Hugging Face CLI: download only (pip install -U huggingface_hub)
hf download LiquidAI/LFM3-1.7B-GGUF --include "*Q4_K_M*.gguf"
```

A shortcut to avoid typing long paths (the rest of this guide uses `$MODEL` and, for vision, `$MMPROJ`):

```bash
export MODEL=/path/to/model.gguf
# e.g. after the `hf download` above:
export MODEL=$(ls ~/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/*/*Q4_K_M*.gguf | head -1)
```

## 2. llama.cpp arguments (reference)

Based on `llama-server --help` and `llama-cli --help` (build 12110). Almost every argument has an equivalent `LLAMA_ARG_*` environment variable (for example `LLAMA_ARG_CTX_SIZE=32768`). To see them all: `llama-server --help`.

> **Defaults worth knowing:** in build 12110, the default `--port` is **9931** (not 8080 as in earlier builds; this guide always passes `--port 8080` explicitly), `--host 127.0.0.1`, `-ngl auto`, `-fa auto`, `--jinja` already on, `-c 0` (uses the model's context, adjusted by `--fit`), `--webui` on and `--slots` on.

### 2.1 Model and download

| Argument | What it does |
|---|---|
| `-m, --model FILE` | Path to the GGUF |
| `-hf, --hf-repo user/repo[:quant]` | Download from Hugging Face (default quant `Q4_K_M`); also fetches the `mmproj` if there is one |
| `-hff FILE` | A specific file from the repo (overrides the quant) |
| `-hft TOKEN` | HF token (or the `HF_TOKEN` env var) |
| `-mu, --model-url URL` | Download from a URL |
| `--offline` | Cache only, no network |
| `-cl, --cache-list` | List the models in the cache |
| `--lora F[,F2]`, `--lora-scaled F:SCALE` | LoRA adapters |
| `-a, --alias NAME` | Model name in the API (the `id` you'll see in `/v1/models` and in omp's `models.yml`) |
| `--tags` | Informational tags |

### 2.2 Context, memory and performance

| Argument | What it does |
|---|---|
| `-c, --ctx-size N` | Total context in tokens (`0` = the model's). Uses RAM for the KV cache |
| `-n, --predict N` | Maximum tokens to generate (`-1` = infinite) |
| `-b, --batch-size N` | Maximum logical batch (2048) |
| `-ub, --ubatch-size N` | Maximum physical batch (512). Larger = faster long prompts, more memory |
| `-ngl, --gpu-layers N\|auto\|all` | Layers on the GPU (Metal). On Apple Silicon with unified memory, `all` |
| `-fa, --flash-attn on\|off\|auto` | Flash attention; reduces memory with long contexts |
| `-ctk / -ctv TYPE` | KV cache type for K/V: `f16` (default), `q8_0`, `q4_0`… `q8_0` saves almost half the memory with little loss |
| `-kvo / -nkvo` | KV cache on the GPU (default) or on the CPU |
| `-t, --threads N` / `-tb` | CPU threads for generation / for prompt processing |
| `--fit on\|off`, `-fitt`, `-fitc` | Automatically adjusts the arguments you didn't set so that everything fits in memory (default margin 1024 MiB) |
| `-lm, --load-mode` | `auto`, `mmap`, `mlock` (keeps the model in RAM), `mmap+mlock`, `dio` |
| `--no-warmup` | Skips the warm-up pass |
| `--keep N` | Tokens of the initial prompt kept when the context overflows |
| `--context-shift` | Shifts the context during infinite generation (off by default) |
| `-cram, --cache-ram N` | Prompt cache in RAM, MiB (8192) |
| `--cache-prompt / --no-cache-prompt` | Reuse the previous prompt (default: yes) |
| `--cache-reuse N` | Minimum chunk size to reuse via KV shifting |
| `-ctxcp N`, `-cms N` | Context checkpoints per slot (32) and their minimum spacing |
| `--list-devices`, `-dev` | Available devices and which ones to use |
| `-ot, --override-tensor`, `-cmoe`, `-ncmoe N` | Place tensors by hand; move MoE expert weights to the CPU (large MoE models) |
| `--rope-scaling`, `--rope-scale`, `--yarn-*` | Extend the context beyond the training length (YaRN/RoPE). Only use it if you need it |

### 2.3 Server (`llama-server`)

| Argument | What it does |
|---|---|
| `--host`, `--port` | Address and port (default `127.0.0.1` and, in build 12110, port 9931) |
| `-np, --parallel N` | Slots (concurrent requests). `-c` is split among them |
| `-cb / -nocb` | Continuous batching (default: yes) |
| `-kvu / -no-kvu` | KV cache unified across slots |
| `--kv-unified-per-slot N` | Context per slot; with `-np 2` and `N=32768`, total KV 65536 |
| `--api-key K[,K2]`, `--api-key-file F` | Requires `Authorization: Bearer ...` |
| `--ssl-key-file`, `--ssl-cert-file` | HTTPS |
| `--webui / --no-webui` | Enables or disables the web UI (`--webui-config JSON` for settings) |
| `--metrics` | Prometheus `/metrics` endpoint |
| `--slots / --no-slots` | `/slots` endpoint (monitoring) |
| `--props` | Allows changing global properties with `POST /props` |
| `--slot-save-path DIR` | Saves/restores each slot's KV cache |
| `-to, --timeout N` | Read/write timeout (3600 s) |
| `--threads-http N` | HTTP threads |
| `--cors-origins`, `--cors-methods`, `--cors-headers` | CORS control (default `*`) |
| `--api-prefix /path` | Serve under a prefix |
| `--sleep-idle-seconds N` | The server "sleeps" (unloads the model) after N s idle |
| `--embedding`, `--rerank`, `--pooling` | Embedding and reranking modes |
| `--models-dir`, `--models-preset`, `--models-max`, `--models-autoload` | **Router** server: serves several models from a directory and loads/unloads them on demand (up to 4 by default) |
| `--spec-default` and shortcuts `--gpt-oss-20b-default`, `--fim-qwen-*-default`, … | Presets that **download** weights from the internet |

**Built-in tools (experimental, dangerous):** `--tools`, `--agent`, `--mcp-servers-config`, `--ui-mcp-proxy` give the model access to files, the shell and MCP *from the server itself*. Don't enable them on servers reachable from the network (`--host 0.0.0.0`). For agents, a client such as omp is a better choice (see [omp.md](omp.md)).

### 2.4 Chat, templates and reasoning

| Argument | What it does |
|---|---|
| `--jinja / --no-jinja` | The model's Jinja chat template (on); required for tool calling |
| `--chat-template NAME`, `--chat-template-file F` | Custom template or a built-in one (`chatml`, `llama3`, `gemma`, …) |
| `--chat-template-kwargs '{"k":"v"}'` | Extra variables for the template |
| `-rea, --reasoning on\|off\|auto` | Turns reasoning on or off |
| `--reasoning-format none\|deepseek\|deepseek-legacy` | Where the thoughts are returned: `reasoning_content` (separate) or inside `content` |
| `--reasoning-effort LEVEL` | `minimal`…`max`, for templates that support it |
| `--reasoning-budget N` | Cap on reasoning tokens (`-1` unlimited, `0` no reasoning) |
| `--reasoning-budget-message` | Text injected when the budget runs out |
| `--reasoning-preserve` | Keeps previous reasoning in the history |
| `--skip-chat-parsing` | Everything (reasoning and tool calls) in `content` |
| `--prefill-assistant` | If the last message is from the assistant, continues it |

### 2.5 Sampling (answer quality and style)

| Argument | Default | Effect |
|---|---|---|
| `--temp` | 0.80 | Randomness: low (0.2-0.5) for code and tools, high for creativity |
| `--top-k` | 40 | Only considers the K most likely tokens |
| `--top-p` | 0.95 | Sampling by cumulative probability mass |
| `--min-p` | 0.05 | Drops tokens below a fraction of the most likely one |
| `--repeat-penalty`, `--repeat-last-n` | 1.0, 64 | Penalizes repetition (>1 reduces it) |
| `--presence-penalty`, `--frequency-penalty` | 0 | Presence and frequency penalties |
| `--dry-multiplier` | 0 | DRY sampling against long repetitions |
| `--mirostat 1\|2` | 0 | Controls entropy instead of top-k/top-p |
| `-s, --seed` | -1 | Random seed; set it for reproducible results |
| `--samplers`, `--sampler-seq` | | Order of the samplers |
| `--grammar`, `--grammar-file` | | BNF grammar to enforce the format |
| `-j, --json-schema`, `-jf FILE` | | Forces JSON output that follows a schema |
| `--ignore-eos` | | Don't stop at the end-of-sequence token |
| `-l, --logit-bias` | | Raises or lowers a token's probability |

The sampling values each model recommends are usually in its Hugging Face model card (e.g. the one for `LiquidAI/LFM3-1.7B-GGUF`).

### 2.6 Multimodal (image, audio, video)

| Argument | What it does |
|---|---|
| `-mm, --mmproj FILE` | Multimodal projector (downloaded automatically with `-hf`) |
| `--no-mmproj` | Don't use the projector |
| `--mmproj-offload / --no-mmproj-offload` | Projector on GPU or CPU |
| `--image / --audio / --video FILE` | **`llama-cli` only**: attaches files (several, comma-separated) |
| `--image-min-tokens`, `--image-max-tokens` | Token limits per image |
| `--video-fps` | Video frames per second (4) |

### 2.7 Speculative decoding (speeds up generation)

| Argument | What it does |
|---|---|
| `-md, --spec-draft-model F` | Small "draft" model of the same family as the main one |
| `--spec-type TYPE` | `draft-simple`, `ngram-simple`, `ngram-mod`, `ngram-cache`, … (default `none`) |
| `--spec-draft-n-max N`, `--spec-draft-n-min N` | Tokens proposed by the draft model (3 by default) |
| `-hfd user/repo` | Draft model from Hugging Face |
| `-ngld N` | Draft model layers on the GPU |

### 2.8 `llama-cli`-specific arguments

| Argument | What it does |
|---|---|
| `-p, --prompt TEXT`, `-f, --file F` | Initial prompt, or from a file |
| `-sys, --system-prompt TEXT`, `-sysf F` | System prompt |
| `-st, --single-turn` | A single turn, then exits (with `-p` it doesn't enter interactive mode) |
| `-mli, --multiline-input` | Lets you paste several lines without `\` |
| `-r, --reverse-prompt TEXT` | Stops generation and hands control back |
| `--no-display-prompt` | Doesn't echo the prompt in the output |
| `-co, --color on\|off\|auto` | Colors the output |
| `--show-timings / --no-show-timings` | Statistics after each answer |
| `--simple-io` | Basic I/O for subprocesses |
| `--server-base URL` | Connects to an already running server instead of loading the model |
| `-sp, --special` | Shows special tokens |
| `--verbose-prompt` | Prints the final prompt with the template applied |

### 2.9 Logging and debugging

`-v` (everything), `-lv N` (level 0-5; default 3), `--log-file F`, `--log-jsonl`, `--log-colors`, `--log-timestamps`, `--log-disable`, `--perf`, `--check-tensors`, `--version`, `--log-prompts-dir DIR` (saves every prompt for debugging).

### 2.10 Useful combinations

```bash
# Local, fast and light on memory (quantized KV cache)
llama-server -m "$MODEL" --alias lfm3-1.7b --port 8080 -c 32768 -ngl all -fa on -ctk q8_0 -ctv q8_0 --jinja

# Two requests in parallel (omp with subagents): 32k of context per slot
llama-server -m "$MODEL" --alias lfm3-1.7b --port 8080 -np 2 -c 65536 -ngl all -fa on --jinja

# Deterministic answers for scripts
llama-server -m "$MODEL" --port 8080 --temp 0 --seed 42

# No reasoning (for models that reason by default, such as LFM3)
llama-server -m "$MODEL" --port 8080 --reasoning off

# Guaranteed JSON output
llama-cli -m "$MODEL" -st -p "Give me 3 colors" -jf schema.json

# Accept connections from the network, with a key
llama-server -m "$MODEL" --host 0.0.0.0 --port 8080 --api-key MY_KEY
```

## 3. llama-cli (interactive chat in the terminal)

```bash
# Chat with a local model
llama-cli -m "$MODEL" -c 8192 -ngl 99 -fa on

# Or download it from Hugging Face
llama-cli -hf LiquidAI/LFM3-1.7B-GGUF:Q4_K_M -c 8192 -ngl 99 -fa on

# A single answer, then exit
llama-cli -m "$MODEL" -ngl 99 -p "Explain what a GGUF is in 2 sentences" -n 200 -st --no-display-prompt

# Vision model with an image (needs its mmproj)
llama-mtmd-cli -m "$MODEL" --mmproj "$MMPROJ" \
  --image photo.jpg -p "Describe the image" -ngl 99
```

In the chat: `Ctrl+C` interrupts generation; `/exit` or a double `Ctrl+C` quits.

## 4. llama-server (OpenAI-compatible API)

Exposes `/v1/chat/completions`, `/v1/models`, `/health`, and a web UI at `http://127.0.0.1:8080`.

### Chat model

```bash
llama-server -m "$MODEL" --alias lfm3-1.7b \
  --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 -fa on --jinja
```

`--alias` sets the name clients will see (`/v1/models`, the `model` field of requests, the `id` in omp's `models.yml`).

### Vision model (`--mmproj`)

```bash
llama-server -m "$MODEL" --mmproj "$MMPROJ" \
  --alias my-vision-model --port 8081 -c 32768 -ngl 99 -fa on --jinja
```

With `-hf`, the `mmproj` is downloaded automatically if the repository includes it. For large models, lower `-c` and don't load another large model at the same time.

### Embedding model

```bash
llama-server -m /path/to/embedding-model.gguf \
  --embedding --alias embed --port 8084 -c 2048 -ngl 99
# curl localhost:8084/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"embed","input":"hello"}'
```

### Download and serve from Hugging Face

```bash
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q4_K_M --alias lfm3-1.7b --port 8080 -c 32768 -ngl 99 --jinja
```

### In the background

```bash
nohup llama-server -m "$MODEL" --alias lfm3-1.7b --port 8080 -c 32768 -ngl 99 -fa on --jinja \
  > ~/llama-server.log 2>&1 &
echo $! > ~/llama-server.pid
# stop
kill $(cat ~/llama-server.pid)     # or:  pkill -f llama-server
```

### Check

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"lfm3-1.7b","messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
```

## 5. Tips and common issues

- **Ports in use**: LM Studio uses `1234`, Ollama `11434`, and `llama-server` without `--port` (build 12110) `9931`. To see what's using a port: `lsof -iTCP:8080 -sTCP:LISTEN`.
- **Memory**: the model takes roughly the size of the `.gguf` file; add the KV cache (it grows with `-c`) and don't load several large models at once. On macOS keep an eye on `Activity Monitor › Memory`; on Linux, `nvidia-smi` or `free -h`.
- **Context vs. `-np`**: with `-np 4 -c 32768` each slot gets 8192 tokens.
- **A reasoning model (e.g. LFM3) returns `content: ""`** if it runs out of `max_tokens` while reasoning; the text is in `reasoning_content`. Raise `max_tokens` or use `--reasoning off`.
- **Tool calling**: requires `--jinja`. A 1.7B model may fail with complex tools.
- **Speed benchmark**: `llama-bench -m "$MODEL" -ngl 99 -fa 1` (for capabilities, see this repo's benchmarks).
- **GGUF info**: `llama-gguf file.gguf r`.
- **Security**: the server listens on `127.0.0.1`. If you use `--host 0.0.0.0`, add `--api-key`.
- **Updating**: `brew upgrade llama.cpp` (or rebuild from the llama.cpp repository).
