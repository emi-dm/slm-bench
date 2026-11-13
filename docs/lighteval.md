# lighteval in slmbench

slmbench is **built on [lighteval](https://github.com/huggingface/lighteval)** (Hugging Face): every evaluation runs through lighteval. lighteval provides the tasks and the metrics; slmbench provides the model (an OpenAI API client that talks to your server), the modes, the TUI, the reports and the leaderboard.

```text
 slmbench run / TUI
   |
   +-- lighteval custom <label> bench/slmbench/lighteval_model.py "<task>|0,..." --custom-tasks bench/slmbench/lighteval_tasks.py
         |                                                  |
         |                                                  +-- custom tasks: IFStruct, HumanEval+, MBPP+, CRUXEval, MultiPL-E, LCB
         +-- SlmbenchOpenAIModel --POST /v1/chat/completions--> your server (llama-server, Ollama, LM Studio, vLLM)
```

## Components

| Component | What it does |
|---|---|
| `bench/slmbench/engine.py` | Launches lighteval as a subprocess, collects results and statistics, writes `report.md` and `summary.json`, updates the leaderboard and marks the run as "running" while it lasts |
| `bench/slmbench/lighteval_model.py` | The model for `lighteval custom`: calls `/v1/chat/completions` with the standard library, sends the task's system prompt if it has one, applies max tokens, the reasoning budget and extra JSON, retries errors and records every request |
| `bench/slmbench/lighteval_tasks.py` | Custom tasks passed with `--custom-tasks` |
| `bench/slmbench/benchmarks/` | The custom tasks (`ifstruct.py` + the vendored official validator, `code.py`, `lcb.py`), the catalog with the modes (`catalog.py`) and the cards with a sample task (`explore.py`) |

Settings reach the model through environment variables that slmbench fills in:

| Variable | Source |
|---|---|
| `SLMBENCH_LD_BASE_URL`, `SLMBENCH_LD_MODEL`, `SLMBENCH_LD_API_KEY` | The model to evaluate |
| `SLMBENCH_LD_MAX_TOKENS` | `--max-tokens` (overrides the task's value; default 2048) |
| `SLMBENCH_LD_REASONING_BUDGET` | `--reasoning-budget` (llama-server: `thinking_budget_tokens`; default 1024) |
| `SLMBENCH_LD_TEMPERATURE`, `SLMBENCH_LD_CONCURRENCY` | `--temperature`, `--workers` |
| `SLMBENCH_LD_TIMEOUT`, `SLMBENCH_LD_MAX_CONTEXT`, `SLMBENCH_LD_EXTRA_BODY` | Settings |
| `SLMBENCH_LD_STATS` | JSONL file where every request is logged (tok/s, empty, truncated, progress) |
| `SLMBENCH_LD_RESPONSES` | JSONL file where every answer is saved (key, problem, task, text, reasoning, finish, tokens): used for rescoring, resuming and the failure explorer |
| `SLMBENCH_LD_REPLAY` | Saved answers of previous runs: an answer whose key (prompt + generation parameters) matches is reused instead of calling the model (`run --resume`) |
| `SLMBENCH_LD_REPLAY_ONLY` | `1`: never call the model; answers not found stay empty (`rescore`) |

## Why a custom model and not `lighteval endpoint litellm`

The first version used the litellm backend. It was replaced because of:

- **Fragility.** It required pinning `litellm==1.103.11` (≥ 1.109 fails at startup: `LiteLLMCacheType.DISK`) and adding `orjson` and `soundfile`; without them every call failed **silently** and the score came out as 0%.
- **Visible errors.** Now, if a request still fails after retrying, the report says "Failed requests: N" with an example.
- **No misleading caches.** Neither lighteval's sample cache nor litellm's is used: every run queries the model.
- **Control.** `max_tokens`, `seed`, `stop`, the reasoning budget and each task's system prompt are sent as they are.

IFEval with 30 samples gave exactly the same result with both integrations (63.3% ± 8.9 with LFM3-1.7B Q4_K_M).

## Pinned versions

lighteval 0.13.0 (Nov 2025) breaks with later versions of its dependencies, so `pyproject.toml` pins them:

| Package | Symptom without the pin |
|---|---|
| `huggingface_hub<1`, `transformers<5`, `datasets<5` | `TypeError: HfApi.list_models() got an unexpected keyword argument 'model_name'` when loading the task registry |
| `xxhash<3.6` | `TypeError: Strings must be encoded before hashing` when saving the details |

If something breaks after an upgrade: `uv tool install --editable . --reinstall` (or `uv sync --reinstall`).

## Limitations

- **Generative tasks only.** Tasks scored by logprobs (classic MMLU, HellaSwag and ARC) don't work against a chat API.
- **The model class depends on lighteval 0.13's internal interface** (for example, the pipeline requires a `_cache` attribute). It has to be reviewed when upgrading lighteval.
- **Gated datasets.** GPQA, for example, requires accepting its terms on Hugging Face and logging in (`hf auth login`).
- **Reasoning models.** If they run out of tokens, the answer is empty and scores 0; the report counts empty and truncated answers.
