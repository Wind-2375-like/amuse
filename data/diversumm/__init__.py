"""Load DiverSumm (Infuse) into the same normalized schema as AggreFact.

Source: https://github.com/HJZnlp/Infuse/blob/main/DiverSumm.csv
The CSV has 563 rows across 5 origins (ChemSum, GovReport, arXiv, multinews,
qmsum), with 3-12 distinct summarizer systems per origin. Unlike LLM-AggreFact,
summaries are full multi-sentence passages (mean 3-15 sentences depending on
origin), so aggregation methods are actually exercised.

Output schema matches data/aggrefact:
    doc_id, document, summary, human_label, score, origin, split, model, cut
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd


DIVERSUMM_URL = "https://raw.githubusercontent.com/HJZnlp/Infuse/main/DiverSumm.csv"


def _doc_id(document: str, summary: str, idx: int) -> str:
    h = hashlib.sha1((document + "||" + summary).encode("utf-8")).hexdigest()[:12]
    return f"diversumm-{idx:06d}-{h}"


def load_diversumm(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load DiverSumm. If `csv_path` is None, download to a tmp location."""
    if csv_path is None:
        cached = Path("/tmp/DiverSumm.csv")
        if not cached.exists():
            print(f"[load_diversumm] downloading {DIVERSUMM_URL} -> {cached}", flush=True)
            urllib.request.urlretrieve(DIVERSUMM_URL, cached)
        csv_path = str(cached)

    raw = pd.read_csv(csv_path)
    expected = {"origin", "id", "doc", "summary", "model_name", "label"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"DiverSumm CSV missing columns: {missing}; got {list(raw.columns)}")

    out = pd.DataFrame()
    out["document"] = raw["doc"].astype(str)
    out["summary"] = raw["summary"].astype(str)
    # label: 1 = faithful, 0 = unfaithful (Infuse paper convention)
    out["human_label"] = pd.to_numeric(raw["label"], errors="coerce").astype("Int64")
    out["score"] = float("nan")
    out["origin"] = raw["origin"].astype(str)
    out["split"] = "test"
    out["model"] = raw["model_name"].astype(str)
    out["cut"] = "DiverSumm"
    out["doc_id"] = [
        _doc_id(d, s, i)
        for i, (d, s) in enumerate(zip(out["document"], out["summary"]))
    ]

    return out[
        ["doc_id", "document", "summary", "human_label", "score",
         "origin", "split", "model", "cut"]
    ]
