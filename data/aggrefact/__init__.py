"""Load AggreFact from HuggingFace into a normalized DataFrame.

Output schema (one row per (doc, summary)):
    doc_id     : str
    document   : str
    summary    : str
    human_label: int      # 1 = faithful, 0 = unfaithful (binary label)
    score      : float    # original continuous label if present, else NaN
    origin     : str      # "cnndm" / "xsum" / ...
    split      : str      # "val" / "test" / ...
    model      : str      # the summarization system that produced `summary`
    cut        : str      # AggreFact-FtSota / AggreFact-CnnDm / etc.

Usage:
    python scripts/load_aggrefact.py --out data/aggrefact/aggrefact.parquet
"""
from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd


# Candidate HF dataset IDs (we try in order; the AggreFact community has a few mirrors).
_CANDIDATES = [
    "yuh-zha/AggreFact",
    "lytang/LLM-AggreFact",  # broader superset; fallback
]


def _doc_id(document: str, summary: str, idx: int) -> str:
    h = hashlib.sha1((document + "||" + summary).encode("utf-8")).hexdigest()[:12]
    return f"aggrefact-{idx:06d}-{h}"


def _normalize(df: pd.DataFrame, source_id: str) -> pd.DataFrame:
    """Map raw HF columns onto our schema. Different mirrors use different
    column names; handle the common ones."""
    cols = {c.lower(): c for c in df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    doc_col = col("doc", "document", "source")
    sum_col = col("summary", "claim", "hypothesis")
    label_col = col("label", "human_label", "is_factual", "faithful")
    score_col = col("score", "human_score", "rating")
    origin_col = col("origin", "dataset", "source_dataset")
    split_col = col("split", "cut")
    model_col = col("model", "model_name", "system")
    cut_col = col("cut", "subset", "aggrefact_cut")

    if doc_col is None or sum_col is None:
        raise ValueError(
            f"Cannot find doc/summary columns in {source_id}. Got: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["document"] = df[doc_col].astype(str)
    out["summary"] = df[sum_col].astype(str)
    if label_col is not None:
        out["human_label"] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    else:
        out["human_label"] = pd.array([pd.NA] * len(df), dtype="Int64")
    out["score"] = (
        pd.to_numeric(df[score_col], errors="coerce") if score_col else float("nan")
    )
    out["origin"] = df[origin_col].astype(str) if origin_col else source_id
    out["split"] = df[split_col].astype(str) if split_col else "unknown"
    out["model"] = df[model_col].astype(str) if model_col else "unknown"
    out["cut"] = df[cut_col].astype(str) if cut_col else source_id

    out["doc_id"] = [_doc_id(d, s, i) for i, (d, s) in enumerate(zip(out["document"], out["summary"]))]

    # Reorder
    out = out[
        ["doc_id", "document", "summary", "human_label", "score",
         "origin", "split", "model", "cut"]
    ]
    return out


def load_aggrefact(
    splits: Optional[tuple[str, ...]] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Try each candidate source until one loads. Returns concatenated frame.

    `splits=None` means "use whatever splits the dataset actually has"
    (recommended; AggreFact mirrors disagree on split names — some only have
    `test`).
    """
    from datasets import (  # lazy import
        load_dataset,
        get_dataset_config_names,
        get_dataset_split_names,
    )

    errors: list[str] = []
    for ds_id in _CANDIDATES:
        print(f"[load_aggrefact] trying {ds_id} ...", flush=True)
        try:
            configs = get_dataset_config_names(ds_id)
        except Exception as e:
            errors.append(f"{ds_id}: get_configs failed: {e!r}")
            configs = [None]
        if not configs:
            configs = [None]

        frames = []
        for cfg in configs:
            # Resolve splits dynamically per config.
            try:
                avail = get_dataset_split_names(ds_id, cfg) if cfg else get_dataset_split_names(ds_id)
            except Exception as e:
                errors.append(f"{ds_id}/{cfg}: get_splits failed: {e!r}")
                avail = []
            if not avail:
                # Fall back to common names.
                avail = ["test", "validation", "train"]

            wanted = list(splits) if splits else avail
            # Keep only splits that actually exist (intersection preserving order).
            wanted = [s for s in wanted if s in avail] or avail

            for split in wanted:
                try:
                    ds = (
                        load_dataset(ds_id, cfg, split=split)
                        if cfg else load_dataset(ds_id, split=split)
                    )
                except Exception as e:
                    errors.append(f"{ds_id}/{cfg}/{split}: load failed: {e!r}")
                    continue
                try:
                    df = ds.to_pandas()
                    df = _normalize(df, source_id=f"{ds_id}/{cfg or 'default'}")
                except Exception as e:
                    errors.append(f"{ds_id}/{cfg}/{split}: normalize failed: {e!r}")
                    continue
                df["split"] = df["split"].where(df["split"].astype(str) != "unknown", split)
                print(
                    f"[load_aggrefact]   loaded {ds_id}/{cfg}/{split}: {len(df)} rows",
                    flush=True,
                )
                frames.append(df)

        if frames:
            full = pd.concat(frames, ignore_index=True)
            if max_rows is not None:
                full = full.head(max_rows).reset_index(drop=True)
            return full

    msg = "\n  ".join(errors) if errors else "(no inner errors collected)"
    raise RuntimeError(
        f"Could not load AggreFact from any of {_CANDIDATES}. Inner errors:\n  {msg}"
    )
