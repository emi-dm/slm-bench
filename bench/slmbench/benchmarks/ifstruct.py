"""IFStruct v1.0 (Liquid AI): structured-output compliance, as a lighteval task.

Each prompt asks for a few items in JSON or YAML following a schema, with constraints phrased in many different ways (item
count, wrapper key, code fence, no commentary, enums, numeric bounds, nested lists, escaping). Only the structure is
judged, and a response passes only if every constraint holds.

Faithful to the official evaluator (https://github.com/Liquid4All/ifstruct):
- the prompt is sent alone as the user message (no system prompt), like `ifstruct/client.py`;
- scoring uses the official validator, vendored unchanged in `ifstruct_validator.py`.

Metrics (averaged over prompts):
- pass_rate       official score: 1 if the response satisfies every constraint, else 0
- parse_rate      the payload parses as the requested format (JSON or YAML)
- schema_match    share of schema checks passed (partial credit, diagnostic only)
- pass_json / pass_yaml   pass_rate restricted to JSON / YAML prompts
"""
from __future__ import annotations

import json
import logging

import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

from slmbench.benchmarks.ifstruct_validator import validate_response

logger = logging.getLogger(__name__)

METRICS = ["pass_rate", "parse_rate", "schema_match", "pass_json", "pass_yaml"]


def _load(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _dump(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def ifstruct_prompt(line: dict, task_name: str = "ifstruct") -> Doc:
    # json_schema and top_level_count stay as JSON text: their shape varies between rows (a count is an int or a range), and
    # lighteval saves `specific` to Parquet, which needs one type per column. They are parsed when scoring.
    spec = {
        "json_schema": _dump(line["json_schema"]),
        "top_level_count": _dump(line["top_level_count"]),
        "top_level_key": line.get("top_level_key") or None,
        "require_wrapper_key": bool(line["require_wrapper_key"]),
        "require_code_block": bool(line["require_code_block"]),
        "require_no_commentary": bool(line["require_no_commentary"]),
        "output_format": line["output_format"],
    }
    return Doc(task_name=task_name, query=line["prompt"], choices=[""], gold_index=0,
               specific={**spec, "entity_type": line.get("entity_type"), "doc_id": line.get("doc_id")})


def score(response: str, spec: dict) -> dict[str, float]:
    keys = ("json_schema", "top_level_count", "top_level_key", "require_wrapper_key", "require_code_block",
            "require_no_commentary", "output_format")
    args = {k: spec[k] for k in keys}
    args["json_schema"], args["top_level_count"] = _load(spec["json_schema"]), _load(spec["top_level_count"])
    v = validate_response(response=response or "", **args)
    fmt = spec["output_format"]
    ok = 1.0 if v.passed else 0.0
    parsed = v.details.get("yaml_valid" if fmt == "yaml" else "json_valid", False)
    return {"pass_rate": ok, "parse_rate": 1.0 if parsed else 0.0,
            "schema_match": float(v.details.get("schema_match_ratio", 0.0)),
            "pass_json": ok if fmt == "json" else float("nan"), "pass_yaml": ok if fmt == "yaml" else float("nan")}


class IFStructScore(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        text = (model_response.final_text or [""])[0] or ""
        try:
            return score(text, doc.specific)
        except Exception as e:                                  # a validator crash never takes the run down
            logger.warning("ifstruct validator error: %r", e)
            fmt = doc.specific.get("output_format")
            return {"pass_rate": 0.0, "parse_rate": 0.0, "schema_match": 0.0,
                    "pass_json": 0.0 if fmt == "json" else float("nan"), "pass_yaml": 0.0 if fmt == "yaml" else float("nan")}


ifstruct_metric = SampleLevelMetricGrouping(
    metric_name=list(METRICS), higher_is_better={m: True for m in METRICS}, category=SamplingMethod.GENERATIVE,
    sample_level_fn=IFStructScore(), corpus_level_fn={m: np.nanmean for m in METRICS}, batched_compute=False)

ifstruct = LightevalTaskConfig(
    name="ifstruct", prompt_function=ifstruct_prompt, hf_repo="LiquidAI/ifstruct-v1.0", hf_subset="default",
    hf_avail_splits=["test"], evaluation_splits=["test"], few_shots_split=None, few_shots_select=None,
    generation_size=16000,          # the official default (--max-tokens 16000); slmbench replaces it with "max tokens per answer"
    metrics=[ifstruct_metric], stop_sequence=[], version=0,
)

TASKS = [ifstruct]
