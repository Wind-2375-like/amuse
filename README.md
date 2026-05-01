# NLP Project — Aggregation Methods for LLM-based Summary Faithfulness Evaluation

**Research question.** When an LLM evaluates summary faithfulness sentence by
sentence, how should those per-sentence judgements be aggregated into a
summary-level score? We compare `min` / `mean` / `max` / weighted aggregators
against human labels on AggreFact (and later other meta-eval benchmarks).

This repo currently implements **Phase 1**: data loading, sentence splitting,
and a cached sentence-level LLM scorer. Aggregation and meta-evaluation come
in later phases.

---

## 1. Onboard in 5 minutes (read this first)

```bash
git clone https://github.com/Wind-2375-like/amuse.git amuse && cd amuse

# 1. Create env (Python >=3.10). Conda or venv both fine.
conda create -n amuse python=3.10 -y && conda activate amuse

# 2. Install deps.
#    Add the `serve` extra ONLY on the box that will run vLLM (it's heavy).
pip install -e .                # client side only
pip install -e '.[serve]'       # server side, needs CUDA

# 3. spaCy model (small, ~12MB).
python -m spacy download en_core_web_sm

# 4. Sanity-check end-to-end with the offline mock evaluator (no GPU needed).
python scripts/score_sentences.py --mock --inline-fallback --limit 3
```

You should see a few sentence-level rows printed, and a JSONL cache file
created under `results/cache/`.

To run for real against a live vLLM server, see §3.

---

## 2. Repo layout

```
nlp_project/
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml            # model, endpoint, prompt version, cache dir
├── data/
│   ├── aggrefact/              # AggreFact loader (HF -> parquet)
│   └── sentences.py            # spaCy sentence splitter w/ offsets
├── prompts/
│   └── sentence_faithfulness_v1.txt
├── evaluators/
│   ├── base.py                 # SentenceEvaluator interface + SentenceScore
│   ├── openai_compat.py        # vLLM/OpenAI-compatible impl + MockEvaluator
│   └── cache.py                # JSONL append-only cache + CachedEvaluator
├── aggregation/                # Phase 2 — placeholder
├── eval/                       # Phase 2/3 — placeholder
├── scripts/
│   ├── serve_vllm.sh           # one-line vLLM launcher
│   ├── load_aggrefact.py       # HF -> data/aggrefact/aggrefact.parquet
│   └── score_sentences.py      # MAIN PIPELINE
└── results/
    └── cache/<model_slug>/<dataset>.jsonl   # see §5
```

---

## 3. Running the LLM (vLLM, OpenAI-compatible)

We treat the LLM as an OpenAI-compatible HTTP server. The default is
`Qwen/Qwen3-8B` in bf16 on a single A100-80GB.

```bash
# On the GPU box (needs `pip install -e '.[serve]'`):
bash scripts/serve_vllm.sh
# -> http://localhost:8000/v1
```

Override defaults via env vars:

```bash
MODEL=Qwen/Qwen3-1.7B PORT=8001 MAX_LEN=4096 bash scripts/serve_vllm.sh
```

Then on the same (or another) machine, point the client at the endpoint:

```bash
ENDPOINT=http://<gpu-host>:8000/v1 \
MODEL_NAME=Qwen/Qwen3-8B \
python scripts/score_sentences.py --limit 20
```

Or via flags:

```bash
python scripts/score_sentences.py \
    --endpoint http://localhost:8000/v1 \
    --model Qwen/Qwen3-8B \
    --limit 20
```

---

## 4. The Phase-1 pipeline

```bash
# Step A — pull AggreFact (once).
python scripts/load_aggrefact.py --out data/aggrefact/aggrefact.parquet

# Step B — sentence-level scoring (cached).
python scripts/score_sentences.py --limit 20
# -> results/sentence_scores.parquet
```

Output schema (parquet):

| col              | type    | meaning                              |
|------------------|---------|--------------------------------------|
| `doc_id`         | str     | from AggreFact loader                |
| `summary_id`     | str     | `{doc_id}::{system_model}`           |
| `sent_idx`       | int     | sentence position in the summary     |
| `sent_text`      | str     | the summary sentence                 |
| `faithful`       | int 0/1 | LLM judgement                        |
| `confidence`     | float   | LLM-reported confidence              |
| `reason`         | str     | LLM short rationale                  |
| `raw_response`   | str     | exact LLM text (for debugging)       |
| `parse_failed`   | bool    | true iff we fell back after retries  |
| `model_name`     | str     | what produced the score              |
| `prompt_version` | str     | which prompt template was used       |
| `human_label`    | int     | gold (passed through from AggreFact) |

`--limit N` truncates to the first `N` (doc, summary) pairs. Use it for smoke
tests; do **not** run full AggreFact in Phase 1.

`--mock` swaps in a deterministic offline `MockEvaluator` (token overlap
heuristic). Useful for debugging the pipeline without burning a GPU.

`--inline-fallback` lets `--mock` runs work even when the dataset parquet is
absent (e.g. on a clean checkout).

---

## 5. Cache (READ THIS — it's the most important part for collaboration)

LLM calls are expensive; we cache aggressively.

**Key.** `(doc_hash, summary_hash, sent_idx, model_name, prompt_version)` —
where `*_hash` is `sha1(text)[:16]`. Different model or different prompt
version => different cache entries. Re-running a script never re-calls the
LLM for already-scored sentences.

**File.** `results/cache/<model_slug>/<dataset>.jsonl`, append-only JSONL.
One LLM call = one line. We never rewrite the file, which makes it safe for
multiple processes appending and easy to merge across teammates.

**Sharing across teammates.**

We commit cache files to git so everyone benefits from each other's
LLM spend. They are append-only and small per line (~1 KB), so as long as
the file stays under ~50 MB, plain git is fine. Once a cache file gets large,
move it to **git LFS**:

```bash
git lfs install
git lfs track "results/cache/**/*.jsonl"
git add .gitattributes
```

To **merge two cache files** safely (e.g. teammate A and teammate B both
appended new entries on a topic branch):

```bash
# Just concatenate; the loader dedupes by key on read.
cat results/cache/Qwen_Qwen3-8B/aggrefact.jsonl.theirs \
    >> results/cache/Qwen_Qwen3-8B/aggrefact.jsonl
```

If two entries share the same key (shouldn't happen with deterministic
`temperature=0.0`, but possible across machines/seeds), the loader keeps the
**last** occurrence.

---

## 6. Switching models

Either edit `configs/default.yaml` or set env vars:

```bash
MODEL_NAME=Qwen/Qwen3-1.7B ENDPOINT=http://localhost:8000/v1 \
    python scripts/score_sentences.py --limit 20
```

`MODEL_NAME` is part of the cache key, so each model gets its own cache file.

---

## 7. Phase plan (for context, not implemented yet)

- **Phase 1 (this commit).** Skeleton + AggreFact + sentence splitting +
  cached sentence-level LLM scorer.
- **Phase 2.** Aggregation (`min`/`mean`/`max`/weighted) + summary-level
  scores + meta-evaluation (correlation with `human_label`) on AggreFact.
- **Phase 3.** Add Frank / DiverSumm benchmarks.
- **Phase 4.** Prompt ablations, confidence-weighted aggregation, model
  scaling sweep (Qwen3 0.6B → 32B).

---

## 8. Known caveats / open items

- **AggreFact mirror choice.** The loader tries `yuh-zha/AggreFact` first,
  then falls back to `lytang/LLM-AggreFact`. If neither HF id is reachable
  in your environment, run on a node with internet access and commit the
  parquet. Field harmonisation is best-effort — sanity-check the columns
  before Phase 2.
- **Login nodes have no GPU.** Smoke testing on CPU is supported via
  `--mock`; real runs must happen on a GPU node (slurm `srun` / `sbatch`).
- **Parse failures.** When the LLM returns non-JSON after retries, we record
  `parse_failed=True` with `faithful=0, confidence=0.0`. Track the rate; if
  it's >2%, revisit the prompt before Phase 2.
