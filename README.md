# NLP Project — Aggregation Methods for LLM-based Summary Faithfulness Evaluation

**Research question.** When an LLM evaluates summary faithfulness sentence by
sentence, how should those per-sentence judgements be aggregated into a
summary-level score? We compare `min` / `mean` / `max` / softmin / trimmed
mean / `prob_all_faithful` against human labels on AggreFact (and later other
meta-eval benchmarks).

**Status.**
- **Phase 1** — data loading, sentence splitting, cached sentence-level LLM
  scorer. ✅ done.
- **Phase 2** — aggregation + meta-evaluation (correlation w/ bootstrap CIs)
  on AggreFact-CNN + AggreFact-XSum, plus a sanity scatter plot. ✅ done.
- **Phase 3 / 4** — additional benchmarks, prompt ablations, model scaling.
  Not yet started.

---

## 1. Onboard in 5 minutes

```bash
git clone https://github.com/Wind-2375-like/amuse.git amuse && cd amuse

# 1. Create env (Python >=3.10). Conda or venv both fine.
conda create -n amuse python=3.10 -y && conda activate amuse

# 2. Install deps.
#    Add the `serve` extra ONLY on the box that will run vLLM (it's heavy).
pip install -e .                # client side only
pip install -e '.[serve]'       # server side, needs CUDA

# 3. spaCy model (small, ~12 MB).
python -m spacy download en_core_web_sm

# 4. Sanity-check end-to-end with the offline mock evaluator (no GPU needed).
python scripts/score_sentences.py --mock --inline-fallback --limit 3
```

You should see a few sentence-level rows printed and a JSONL cache file
created under `results/cache/`.

To run for real against a live vLLM server, see §3.

---

## 2. Repo layout

```
nlp_project/
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml              # model, endpoint, prompt version, cache dir
├── data/
│   ├── aggrefact/                # AggreFact loader (HF -> parquet)
│   └── sentences.py              # spaCy sentence splitter w/ offsets
├── prompts/
│   └── sentence_faithfulness_v1.txt
├── evaluators/
│   ├── base.py                   # SentenceEvaluator interface + SentenceScore
│   ├── openai_compat.py          # vLLM/OpenAI-compatible impl + MockEvaluator
│   └── cache.py                  # JSONL append-only cache + CachedEvaluator
├── aggregation/
│   └── methods.py                # min/mean/max/softmin/trimmed/prob_all + registry
├── eval/
│   ├── metrics.py                # pearson/spearman/kendall/roc_auc + bootstrap_ci
│   └── run_meta_eval.py          # MAIN PHASE-2 PIPELINE
├── scripts/
│   ├── serve_vllm.sh             # one-line vLLM launcher
│   ├── load_aggrefact.py         # HF -> data/aggrefact/aggrefact.parquet
│   ├── score_sentences.py        # PHASE-1 PIPELINE
│   └── plot_agg_vs_human.py      # sanity scatter (mean vs min)
└── results/
    ├── cache/<model_slug>/<dataset>.jsonl   # see §6
    ├── sentence_scores.<model_slug>.parquet # Phase 1 output
    ├── summary_scores.<model_slug>.parquet  # Phase 2 per-summary aggregates
    ├── meta_eval_summary.<model_slug>.csv   # Phase 2 long-format results
    ├── meta_eval_table.<model_slug>.md      # Phase 2 human-readable tables
    └── figs/mean_vs_min_scatter.png         # Phase 2 sanity plot
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

## 4. Phase 1 — sentence-level scoring

```bash
# Step A — pull AggreFact (once). lytang/LLM-AggreFact is gated; run
# `huggingface-cli login` first.
python scripts/load_aggrefact.py --out data/aggrefact/aggrefact.parquet

# Step B — sentence-level scoring (cached, defaults to AggreFact-CNN+XSum).
python scripts/score_sentences.py
# -> results/sentence_scores.Qwen_Qwen3-8B.parquet  (default config)
```

The default `--origin AggreFact-CNN AggreFact-XSum` filter is what produces
the canonical 2,352-summary / 4,654-sentence working set. Pass
`--origin all` to disable filtering.

Output schema (`results/sentence_scores.<model_slug>.parquet`):

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
| `origin`         | str     | AggreFact subset (CNN / XSum / ...)  |
| `split`          | str     | dataset split                        |
| `human_label`    | int     | gold (passed through from AggreFact) |

`--limit N` truncates to the first `N` (doc, summary) pairs (smoke testing).
`--mock` swaps in a deterministic offline `MockEvaluator` (token-overlap
heuristic). `--inline-fallback` lets `--mock` runs work even when the dataset
parquet is absent.

---

## 5. Phase 2 — aggregation + meta-evaluation

### 5.1 Run

```bash
# Reads one sentence-score parquet + data/aggrefact/aggrefact.parquet.
# If multiple sentence-score files exist, pass --sentence-scores explicitly.
python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.Qwen_Qwen3-8B.parquet

# -> results/{summary_scores.Qwen_Qwen3-8B.parquet,
#             meta_eval_summary.Qwen_Qwen3-8B.csv,
#             meta_eval_table.Qwen_Qwen3-8B.md}

# Sanity scatter.
python scripts/plot_agg_vs_human.py \
  --summary-parquet results/summary_scores.Qwen_Qwen3-8B.parquet
# -> results/figs/mean_vs_min_scatter.Qwen_Qwen3-8B.png
```

The whole Phase-2 pipeline finishes in well under a minute on a laptop —
no GPU and no LLM calls (everything is read from the sentence-score parquet).

### 5.2 Aggregation methods (`aggregation/methods.py`)

Each function maps a 1-D vector of sentence-level scores to a summary-level
scalar. The registry `AGGREGATIONS` is what `run_meta_eval.py` iterates over.

| name                      | formula                                                    | needs soft input? |
|---------------------------|------------------------------------------------------------|:-:|
| `min`                     | `np.min(s)`                                                | no |
| `mean`                    | `np.mean(s)`                                               | no |
| `max`                     | `np.max(s)`                                                | no |
| `trimmed_mean@0.2`        | mean of `s` after dropping the lowest `floor(0.2·n)` values| no |
| `softmin@tau=0.1/0.5/1.0` | `-tau · logsumexp(-s/tau)` (numerically stable, scipy)     | yes |
| `prob_all_faithful`       | `exp(sum(log(clip(s, 1e-6, 1))))` ≡ `∏ pᵢ`                 | yes |

Two **input semantics** are evaluated and reported separately:
- **`hard`** — uses the binary `faithful` column directly.
- **`soft`** — uses `P(faithful=1) = faithful·confidence + (1-faithful)·(1-confidence)`.

`softmin` and `prob_all_faithful` are only meaningful on soft inputs.
File-level self-tests live at the bottom of `aggregation/methods.py` and
`eval/metrics.py`; run either file directly to execute them.

### 5.3 Metrics (`eval/metrics.py`)

`pearson`, `spearman`, `kendall` (scipy.stats), and `roc_auc` (sklearn).
Each metric is paired with a **percentile bootstrap CI** (`bootstrap_ci`,
1000 reps by default, `seed=0`, `alpha=0.05`).

### 5.4 Outputs

- `results/summary_scores.<model_slug>.parquet` — one row per summary, one column per
  `(aggregation, semantic)` pair, plus `human_label`, `origin`, `n_sent`.
  Used by the plot script and as the entry point for any downstream analysis.
- `results/meta_eval_summary.<model_slug>.csv` — long-format table with columns
  `aggregation, semantic, origin, metric, value, ci_lo, ci_hi, n`.
- `results/meta_eval_table.<model_slug>.md` — human-readable pivot tables, one block per
  origin (`AggreFact-CNN`, `AggreFact-XSum`, `__overall__`), each split into
  `hard` and `soft` sub-tables. CIs that include 0 are flagged with `⁰`.

### 5.5 Headline numbers (Qwen3-8B, AggreFact-CNN + AggreFact-XSum)

Spearman ρ with human label, overall (n = 2352), 95 % bootstrap CI:

| aggregation         | hard                          | soft                          |
|---------------------|-------------------------------|-------------------------------|
| `min`               | +0.477 [+0.435, +0.514]       | +0.448 [+0.409, +0.484]       |
| `mean`              | +0.491 [+0.451, +0.526]       | +0.429 [+0.393, +0.465]       |
| `max`               | +0.493 [+0.454, +0.532]       | +0.454 [+0.419, +0.489]       |
| `trimmed_mean@0.2`  | **+0.494 [+0.455, +0.530]**   | +0.431 [+0.394, +0.467]       |
| `softmin@tau=0.1`   | —                             | +0.276 [+0.235, +0.319]       |
| `softmin@tau=0.5`   | —                             | −0.024 [−0.064, +0.017] ⁰     |
| `softmin@tau=1.0`   | —                             | −0.176 [−0.211, −0.137]       |
| `prob_all_faithful` | —                             | +0.319 [+0.280, +0.363]       |

Best overall Spearman: `trimmed_mean@0.2` on hard inputs (ρ = +0.494).
See `results/meta_eval_table.Qwen_Qwen3-8B.md` for Pearson / Kendall / ROC-AUC and the
per-origin breakdown.

### 5.6 Caveats from the Phase-2 run

- **AggreFact-XSum is essentially single-sentence.** 99.9 % of XSum
  summaries have exactly one sentence under our spaCy split, so every
  aggregation collapses to the same value on XSum. Aggregation choice
  effectively only matters on AggreFact-CNN (mean 3.3 sentences/summary).
  This motivates adding a multi-sentence benchmark in Phase 3.
- **The "min beats mean" hypothesis is only weakly supported.** On hard
  inputs `mean` ≥ `min` overall (the prior); on soft inputs `min` > `mean`.
  CIs overlap heavily — the gap is not significant on AggreFact alone.
- **`softmin` is unnormalized.** With τ ≥ 0.5 it becomes dominated by the
  number of sentences (longer summaries get a more negative score),
  producing the negative correlations above. A length-normalized variant
  (`-τ · (logsumexp(-s/τ) − log N)`) is on the to-do list before the poster.

---

## 6. Cache (READ THIS — it's the most important part for collaboration)

LLM calls are expensive; we cache aggressively.

**Key.** `(doc_hash, summary_hash, sent_idx, model_name, prompt_version)` —
where `*_hash` is `sha1(text)[:16]`. Different model or different prompt
version ⇒ different cache entries. Re-running a script never re-calls the
LLM for already-scored sentences.

**File.** `results/cache/<model_slug>/<dataset>.jsonl`, append-only JSONL.
One LLM call = one line. We never rewrite the file, which makes it safe for
multiple processes appending and easy to merge across teammates.

**Sharing across teammates.** We commit cache files to git so everyone
benefits from each other's LLM spend. Append-only and ~1 KB/line, so plain
git is fine until the file grows past ~50 MB. After that, switch to git LFS:

```bash
git lfs install
git lfs track "results/cache/**/*.jsonl"
git add .gitattributes
```

To **merge two cache files** safely (e.g. teammates A and B both appended
new entries on a topic branch):

```bash
# Just concatenate; the loader dedupes by key on read.
cat results/cache/Qwen_Qwen3-8B/aggrefact.jsonl.theirs \
    >> results/cache/Qwen_Qwen3-8B/aggrefact.jsonl
```

If two entries share the same key (shouldn't happen with `temperature=0.0`,
but possible across machines/seeds), the loader keeps the **last** one.
`parse_failed=True` cache entries are intentionally retried on the next run.

---

## 7. Switching models

Either edit `configs/default.yaml` or set env vars:

```bash
MODEL_NAME=Qwen/Qwen3-1.7B ENDPOINT=http://localhost:8000/v1 \
    python scripts/score_sentences.py --limit 20
```

`MODEL_NAME` is part of the cache key, so each model gets its own cache file.
The default parquet / CSV / markdown outputs are also model-specific, so 4B
and 8B runs no longer overwrite each other.

---

## 8. Phase plan

- **Phase 1.** ✅ Skeleton + AggreFact + sentence splitting + cached
  sentence-level LLM scorer.
- **Phase 2.** ✅ Aggregation registry (`min`/`mean`/`max`/`trimmed_mean`/
  `softmin`/`prob_all_faithful`) + meta-evaluation (Pearson/Spearman/Kendall/
  ROC-AUC with bootstrap CIs) on AggreFact-CNN + AggreFact-XSum.
- **Phase 3.** Add a multi-sentence faithfulness benchmark (e.g. Frank,
  DiverSumm) so aggregation actually has something to aggregate on XSum-like
  data. Add length-normalized `softmin`.
- **Phase 4.** Prompt ablations, confidence-weighted aggregation, model
  scaling sweep (Qwen3 0.6B → 32B).

---

## 9. Known caveats / open items

- **AggreFact mirror choice.** The loader tries `lytang/LLM-AggreFact` first
  (gated; run `huggingface-cli login`), then falls back to `yuh-zha/AggreFact`.
  If neither is reachable from your node, run the loader on a node with
  internet access and commit the parquet.
- **Login nodes have no GPU.** Smoke testing on CPU is supported via
  `--mock`; real runs must happen on a GPU node (slurm `srun` / `sbatch`).
- **Parse failures.** When the LLM returns non-JSON after retries, we record
  `parse_failed=True` with `faithful=0, confidence=0.0`. The cache layer
  retries those entries on subsequent runs. Track the rate; if it's >2 %,
  revisit the prompt before adding new aggregations.
- **XSum summaries are mostly one sentence.** See §5.6 — aggregation choice
  is only informative on AggreFact-CNN under the current spaCy splitter.
