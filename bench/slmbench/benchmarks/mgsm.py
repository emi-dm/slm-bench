"""MGSM in Spanish (juletxara/mgsm, config "es"): 250 grade-school maths problems translated by humans (Shi et al., 2022).

Prompt in the benchmark's native chain-of-thought format ("Pregunta: … / Respuesta paso a paso:") plus an instruction to end
with the number. Scored like the official evaluation: the last number in the answer must equal `answer_number`.
(lighteval's `mgsm:es` compares the whole answer with exact match, which scores ~0 for a chat model that reasons first.)
"""
from __future__ import annotations

import re

import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

INSTRUCTION = ("Resuelve el siguiente problema razonando paso a paso. Termina tu respuesta con una línea "
               "«La respuesta es N.», donde N es el resultado numérico.")
_NUM = re.compile(r"-?\d[\d.,]*")
_FINAL = re.compile(r"respuesta\s+(?:final\s+)?es\s*:?\s*\**\s*(-?\d[\d.,]*)", re.I)


def mgsm_es_prompt(line: dict, task_name: str = "mgsm_es") -> Doc:
    query = f"{INSTRUCTION}\n\nPregunta: {line['question']}\nRespuesta paso a paso:"
    return Doc(task_name=task_name, query=query, choices=[str(line["answer_number"])], gold_index=0,
               specific={"answer": int(line["answer_number"])})


def parse_number(text: str) -> float | None:
    """The final answer: "La respuesta es N" if present, otherwise the last number (official MGSM criterion).
    Handles thousands separators (1.234 / 1,234) and decimal commas."""
    m = _FINAL.findall(text or "")
    raw = m[-1] if m else (_NUM.findall(text or "") or [None])[-1]
    if raw is None:
        return None
    raw = raw.rstrip(".,")
    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", raw):          # 1.234 or 1,234,567 -> thousands separators
        raw = raw.replace(".", "").replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


class LastNumberMatch(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        got = parse_number((model_response.final_text or [""])[0])
        return 1.0 if got is not None and abs(got - doc.specific["answer"]) < 1e-6 else 0.0


accuracy = SampleLevelMetric(metric_name="accuracy", higher_is_better=True, category=SamplingMethod.GENERATIVE,
                             sample_level_fn=LastNumberMatch(), corpus_level_fn=np.mean, batched_compute=False)

mgsm_es = LightevalTaskConfig(
    name="mgsm_es", prompt_function=mgsm_es_prompt, hf_repo="juletxara/mgsm", hf_subset="es",
    hf_avail_splits=["train", "test"], evaluation_splits=["test"], few_shots_split=None, few_shots_select=None,
    generation_size=2048, metrics=[accuracy], stop_sequence=[], version=0,
)

TASKS = [mgsm_es]
