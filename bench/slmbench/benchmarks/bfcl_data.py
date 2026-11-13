"""BFCL data: where the prepared datasets live and how they are downloaded. Kept free of lighteval (slow to import) so the
TUI and the catalog can check whether BFCL is prepared instantly."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
CATEGORIES = {"simple": 400, "multiple": 200, "parallel": 200, "parallel_multiple": 200}


def data_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "slmbench" / "datasets" / "bfcl"


def prepared(category: str) -> bool:
    return (data_dir() / category / "test.jsonl").exists()


def prepare(on_line=print) -> bool:
    """Downloads BFCL's original question and answer files and writes one local dataset per category."""
    from huggingface_hub import hf_hub_download
    ok = True
    for cat, n in CATEGORIES.items():
        try:
            qf = hf_hub_download(REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
            af = hf_hub_download(REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")
            answers = {}
            for line in open(af, encoding="utf-8"):
                if line.strip():
                    a = json.loads(line)
                    answers[a["id"]] = a["ground_truth"]
            out = data_dir() / cat
            out.mkdir(parents=True, exist_ok=True)
            rows = 0
            with open(out / "test.jsonl", "w", encoding="utf-8") as f:
                for line in open(qf, encoding="utf-8"):
                    if not line.strip():
                        continue
                    q = json.loads(line)
                    f.write(json.dumps({"id": q["id"], "category": cat, "question": json.dumps(q["question"]),
                                        "functions": json.dumps(q["function"]), "answer": json.dumps(answers[q["id"]])}) + "\n")
                    rows += 1
            on_line(f"    ✔ BFCL {cat}: {rows} ({n} expected)")
            ok &= rows == n
        except Exception as e:                                     # noqa: BLE001
            on_line(f"    ✘ BFCL {cat}: {type(e).__name__}: {e}")
            ok = False
    return ok
