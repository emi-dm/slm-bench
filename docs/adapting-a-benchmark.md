# Adapting a Hugging Face benchmark to slmbench

slmbench runs everything through [lighteval](https://github.com/huggingface/lighteval). If a benchmark already exists in lighteval, adding it to the catalog is enough. If it doesn't, you write a **lighteval task**: a bit of Python that says how to turn each dataset row into a prompt and how to score the answer. This guide covers both cases with real examples from the repo.

```text
 HF dataset               lighteval task                              slmbench
 +--------------+   +---------------------------+   +-----------------------------------------+
 | row (dict)   |-->| prompt_function(row)->Doc |-->| lighteval_model.py  -> your server (API)|
 +--------------+   | metric(answer, Doc)       |<--| model answer                            |
                    +---------------------------+   +-----------------------------------------+
                                 |
                                 v
              catalog (mode, main metric) -> TUI, reports, leaderboard
```

## 0. Is it already in lighteval?

Look the benchmark up in the [Open Benchmark Index](https://huggingface.co/spaces/OpenEvals/open_benchmark_index) or with `lighteval tasks inspect <name>`. If it exists and is **generative** (the model writes an answer), it works as is:

```bash
slmbench run --extra "aime24|0"          # try it without changing anything
```

Tasks scored by **likelihood** (MMLU, HellaSwag and ARC in their classic form, which compare the logprobs of the options) don't work against a chat API: you need a generative version.

To make it show up in the TUI, in a mode and in the leaderboard, add it to the catalog (step 4) with `"builtin": True`.

## 1. Look at the dataset

You need the repo name, the configuration (subset), the evaluation split and the fields. Hugging Face's public API gives you all of that without downloading anything:

```bash
curl -s "https://datasets-server.huggingface.co/info?dataset=LiquidAI/ifstruct-v1.0" | python3 -m json.tool | head -40
curl -s "https://datasets-server.huggingface.co/first-rows?dataset=LiquidAI/ifstruct-v1.0&config=default&split=test" | python3 -m json.tool | head -60
```

Also read the dataset card and, if there is one, the official evaluation code: **replicate its prompt and its metric** so your numbers are comparable with the published ones. That's what was done for IFStruct: the prompt is sent exactly as the official client sends it, and scoring uses the official validator, vendored unchanged (it's Apache-2.0).

## 2. Write the task

Create a module in `bench/slmbench/benchmarks/` (for example `my_bench.py`) with three pieces.

### 2.1 `prompt_function`: from row to `Doc`

```python
from lighteval.tasks.requests import Doc

def my_prompt(line: dict, task_name: str = "my_bench") -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",      # what the model receives as the user message
        choices=[line["answer"]],                             # reference answer (optional for generative tasks)
        gold_index=0,
        specific={                                            # whatever your metric needs
            "system_prompt": "Answer briefly.",               # slmbench sends it as the system message
            "answer": line["answer"],
        },
    )
```

- `query` is the user message. If the benchmark has a system prompt, put it in `specific["system_prompt"]`: the slmbench model sends it as a `system` message (IFStruct has none: its prompt goes alone as the user message).
- If a dataset field is JSON stored as text (as with `json_schema` and `top_level_count` in IFStruct), call `json.loads` here.

### 2.2 Metric: from answer to score

```python
import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.requests import SamplingMethod

class ExactAnswer(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs) -> float:
        text = (model_response.final_text or [""])[0]          # final answer (without the reasoning)
        return float(doc.specific["answer"].strip().lower() in text.lower())

exact = SampleLevelMetric(metric_name="accuracy", higher_is_better=True, category=SamplingMethod.GENERATIVE,
                          sample_level_fn=ExactAnswer(), corpus_level_fn=np.mean, batched_compute=False)
```

- A metric returns one number per sample and lighteval averages them (`corpus_level_fn`).
- If you want several figures at once (like IFStruct: *pass rate*, *parse rate*, *schema match*…), use `SampleLevelMetricGrouping`, return a `dict` and pass one `corpus_level_fn` per metric (see `benchmarks/ifstruct.py`).
- **Code**: never run what the model writes on the host. Use `slmbench.sandbox.execute(files, steps)` (or `slmbench.checks.run_python(code, tests, timeout)` for Python), which runs it inside the slmbench sandbox: a Docker Sandboxes microVM with no network and no access to your files. `benchmarks/code.py` has runners for JavaScript, C++, Java, Rust and Go built on it, and `benchmarks/lcb.py` shows how to run an existing test harness (lighteval's) inside the sandbox. Mark the benchmark with mode `code` in the catalog so slmbench refuses to run it without the sandbox.

### 2.3 Task configuration

```python
from lighteval.tasks.lighteval_task import LightevalTaskConfig

my_bench = LightevalTaskConfig(
    name="my_bench",                     # invoked as "my_bench|0"
    prompt_function=my_prompt,
    hf_repo="org/dataset",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None, few_shots_select=None,
    generation_size=2048,                # slmbench replaces it with "max tokens per answer"
    metrics=[exact],
    stop_sequence=[],
    version=0,
)

TASKS = [my_bench]
```

## 3. Register it

In `bench/slmbench/lighteval_tasks.py` (the file slmbench passes to lighteval with `--custom-tasks`):

```python
from slmbench.benchmarks import my_bench
TASKS_TABLE += list(my_bench.TASKS)
```

## 4. Add it to the catalog

In `bench/slmbench/benchmarks/catalog.py`, inside `GENERAL` or `CODE` (or create a new mode in `MODES`):

```python
"my_bench": {
    "name": "My benchmark", "spec": "my_bench|0", "hf": "org/dataset:default/test", "n": 1234,
    "metric": "accuracy",               # main metric: the one shown in reports and the leaderboard
    "builtin": False, "requires": [],   # e.g. ["node"] if it needs an executable
    "link": "https://arxiv.org/abs/…",
    "desc_es": "Qué mide, en una frase.", "desc_en": "What it measures, in one sentence.",
},
```

With that, it shows up in `slmbench benchmarks`, in the TUI (*Benchmarks* and *Run benchmarks*), in `slmbench setup --mode …` (which downloads the dataset) and in the leaderboard. `desc_es` is the Spanish description shown when the TUI is in Spanish.

## 5. Check the scoring before using a model

Run the **reference answers** through your metric: they should score ~100%. If they don't, the bug is in the extraction or in the runner, not in the model. That's how the coding benchmarks were validated (IFStruct has no reference answers, so it was validated by running the upstream validator tests and checking the dataset against the official one):

```python
from datasets import load_dataset
from slmbench.benchmarks.my_bench import my_prompt, ExactAnswer
from lighteval.models.model_output import ModelResponse

ds = load_dataset("org/dataset", "default", split="test").select(range(50))
ok = 0
for row in ds:
    doc = my_prompt(row)
    resp = ModelResponse(text=[row["answer"]])          # "perfect model"
    ok += ExactAnswer().compute(resp, doc)
print(ok / len(ds))                                     # ≈ 1.0
```

Also check that a wrong answer scores 0.

## 6. Run it

```bash
slmbench benchmarks my_bench               # card and real prompt of the first task
slmbench run --benchmarks my_bench --max-samples 10
```

In the report, look at the **empty** and **truncated** answers. If there are many, the model is running out of tokens while reasoning: raise `--max-tokens` or limit reasoning with `--reasoning-budget`. If everything looks right, regenerate the docs:

```bash
slmbench benchmarks --markdown > docs/benchmarks.md
```

## Examples in the repo

| File | What it shows |
|---|---|
| `bench/slmbench/benchmarks/ifstruct.py` | JSON fields stored as text, official validator vendored unchanged, multi-figure metric with per-format slices |
| `bench/slmbench/benchmarks/lcb.py` | Reusing a lighteval task (prompt, dataset, runner) with a different metric: pass@1/pass@5 over n answers |
| `bench/slmbench/benchmarks/code.py` | Running code in Python and other languages, extracting the code block, `requires` |
| `lighteval/tasks/tasks/lcb/main.py` (inside lighteval) | A lighteval task that executes code |
