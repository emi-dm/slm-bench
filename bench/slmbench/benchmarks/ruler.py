"""RULER needle-in-a-haystack (long context), from the fixed RULER set published as simonjegou/ruler (kvpress).

Uses the 8 NIAH tasks (niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue, niah_multiquery; 500 each) at 4k and 8k
tokens (lengths measured with the Llama-3 tokenizer: other tokenizers produce more, so give the server ≥ 1.5× context
per slot). Scored like RULER (scripts/eval/synthetic/constants.py, string_match_all): the share of the expected values
that appear in the answer, case-insensitive. A chat API can't prefill RULER's answer prefix, so it is shown in the prompt.
"""
from __future__ import annotations

import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

NIAH = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery")


def _is_niah(row: dict) -> bool:
    return row.get("task") in NIAH


def ruler_prompt(line: dict, task_name: str = "ruler") -> Doc:
    query = f"{line['context'].rstrip()}\n\n{line['question'].strip()}\nAnswer in one short sentence that starts with: \"{line['answer_prefix'].strip()}\""
    answers = [str(a) for a in line["answer"]]
    return Doc(task_name=task_name, query=query, choices=[", ".join(answers)], gold_index=0,
               specific={"answers": answers, "ruler_task": line["task"]})


class StringMatchAll(SampleLevelComputation):
    """RULER's string_match_all: fraction of the reference values contained in the prediction (case-insensitive)."""

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        pred = ((model_response.final_text or [""])[0] or "").lower()
        refs = doc.specific["answers"]
        return sum(r.lower() in pred for r in refs) / len(refs) if refs else 0.0


string_match_all = SampleLevelMetric(metric_name="string_match_all", higher_is_better=True, category=SamplingMethod.GENERATIVE,
                                     sample_level_fn=StringMatchAll(), corpus_level_fn=np.mean, batched_compute=False)


def _task(name: str, length: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name, prompt_function=ruler_prompt, hf_repo="simonjegou/ruler", hf_subset=length, hf_filter=_is_niah,
        hf_avail_splits=["test"], evaluation_splits=["test"], few_shots_split=None, few_shots_select=None,
        generation_size=256, metrics=[string_match_all], stop_sequence=[], version=0,
    )


TASKS = [_task("ruler_niah_4k", "4096"), _task("ruler_niah_8k", "8192")]
