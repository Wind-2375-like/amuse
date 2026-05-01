"""Sentence splitting utilities (spaCy-based, with offsets)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

_NLP = None  # lazy-loaded spaCy pipeline


def _load_nlp(model: str = "en_core_web_sm"):
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy
    try:
        _NLP = spacy.load(model, disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not installed. Run:\n"
            f"    python -m spacy download {model}"
        ) from e
    return _NLP


@dataclass
class Sentence:
    idx: int
    text: str
    start: int  # char offset into the original summary
    end: int


def split_sentences(text: str, model: str = "en_core_web_sm") -> List[Sentence]:
    if not text or not text.strip():
        return []
    nlp = _load_nlp(model)
    doc = nlp(text)
    out: List[Sentence] = []
    for i, sent in enumerate(doc.sents):
        s = sent.text.strip()
        if not s:
            continue
        out.append(Sentence(idx=len(out), text=s, start=sent.start_char, end=sent.end_char))
    return out
