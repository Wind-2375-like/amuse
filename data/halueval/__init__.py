"""Load HaluEval summarization into the normalized AMUSE schema.

Source: https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json

The original HaluEval summarization release provides 10K source records, each
with one source document and two summaries:
- `right_summary`: faithful reference-like summary
- `hallucinated_summary`: intentionally hallucinated summary

AMUSE expands each source record into two rows so downstream scoring and
meta-evaluation can treat it like the other sentence-faithfulness benchmarks.

Output schema matches data/aggrefact:
    doc_id, document, summary, human_label, score, origin, split, model, cut
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd


HALUEVAL_SUMMARIZATION_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"
    "summarization_data.json"
)


def _doc_id(document: str, summary: str, idx: int) -> str:
    h = hashlib.sha1((document + "||" + summary).encode("utf-8")).hexdigest()[:12]
    return f"halueval-{idx:06d}-{h}"


def load_halueval(json_path: Optional[str] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load HaluEval summarization and normalize it."""
    if json_path is None:
        cached = Path("/tmp/HaluEval_summarization_data.json")
        if not cached.exists():
            print(
                f"[load_halueval] downloading {HALUEVAL_SUMMARIZATION_URL} -> {cached}",
                flush=True,
            )
            urllib.request.urlretrieve(HALUEVAL_SUMMARIZATION_URL, cached)
        json_path = str(cached)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = f.read().strip()

    if not payload:
        raise ValueError(f"HaluEval file is empty: {json_path}")

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in payload.splitlines() if line.strip()]

    if not isinstance(raw, list):
        raise ValueError(f"Expected HaluEval records list, got {type(raw).__name__}")
    if max_rows is not None:
        raw = raw[:max_rows]

    rows = []
    for item_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Row {item_idx} is not an object: {type(item).__name__}")

        missing = {"document", "right_summary", "hallucinated_summary"} - set(item.keys())
        if missing:
            raise ValueError(
                f"HaluEval row {item_idx} missing columns: {sorted(missing)}; got {sorted(item.keys())}"
            )

        document = str(item["document"])
        for field_name, label, model_name in [
            ("right_summary", 1, "reference"),
            ("hallucinated_summary", 0, "hallucinated"),
        ]:
            rows.append(
                {
                    "document": document,
                    "summary": str(item[field_name]),
                    "human_label": label,
                    "score": float("nan"),
                    "origin": "HaluEval-summarization",
                    "split": "test",
                    "model": model_name,
                    "cut": "HaluEval",
                }
            )

    out = pd.DataFrame(rows)
    out["human_label"] = pd.to_numeric(out["human_label"], errors="coerce").astype("Int64")
    out["doc_id"] = [
        _doc_id(document, summary, idx)
        for idx, (document, summary) in enumerate(zip(out["document"], out["summary"]))
    ]

    return out[
        [
            "doc_id",
            "document",
            "summary",
            "human_label",
            "score",
            "origin",
            "split",
            "model",
            "cut",
        ]
    ]