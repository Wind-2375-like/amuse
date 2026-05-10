# NLP Project — Aggregation Methods for Understanding Summary-Faithfulness Evaluation

AMUSE: **A**ggregation **M**ethods for **U**nderstanding **S**ummary-Faithfulness **E**valuation

**Research question.** When an LLM evaluates summary faithfulness sentence by sentence, how should those per-sentence judgements be aggregated into a summary-level score? We compare `min` / `mean` / `max` / softmin / trimmed mean / `prob_all_faithful` against human labels on AggreFact and other meta-eval benchmarks.

---

## 1. Setup

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
│   ├── aggrefact/                # AggreFact loader (HF -> parquet, all 11 origins)
│   ├── aggrefact_cnn/            # AggreFact-CNN parquet (built locally)
│   ├── diversumm/                # DiverSumm loader (Infuse CSV -> parquet)
│   ├── aggrefact_other_multi/    # AggreFact-other-≥2s parquet (built locally)
│   ├── halueval/                 # HaluEval summarization parquet (built locally)
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
│   ├── serve_vllm.sh                  # one-line vLLM launcher
│   ├── load_aggrefact.py              # HF -> data/aggrefact/aggrefact.parquet
│   ├── build_aggrefact_cnn.py         # filter aggrefact to origin == AggreFact-CNN
│   ├── load_diversumm.py              # GitHub CSV -> data/diversumm/diversumm.parquet
│   ├── build_aggrefact_other_multi.py # filter aggrefact to non-CNN, n_sents>=2
│   ├── build_halueval.py              # HaluEval JSON -> data/halueval/halueval.parquet
│   ├── score_sentences.py             # PHASE-1 PIPELINE
│   └── plot_agg_vs_human.py           # sanity scatter (mean vs min)
└── results/
    ├── cache/<model_slug>/<dataset_name>.jsonl       # see §6
    ├── sentence_scores.<model_slug>.<dataset_name>.parquet  # Phase 1 output
    ├── summary_scores.<model_slug>.parquet           # Phase 2 per-summary aggregates
    ├── meta_eval_summary.<model_slug>.csv            # Phase 2 long-format results
    ├── meta_eval_table.<model_slug>.md               # Phase 2 human-readable tables
    └── figs/mean_vs_min_scatter.png                  # Phase 2 sanity plot
```

---

## 3. Serving the LLM

The default is `Qwen/Qwen3-8B` in bf16 on a single A100-80GB with a 16K context window.

```bash
# On the GPU box (needs `pip install -e '.[serve]'`):
bash scripts/serve_vllm.sh
# -> http://localhost:8000/v1
```

Override defaults via env vars:

```bash
MODEL=Qwen/Qwen3-1.7B PORT=8001 MAX_LEN=4096 bash scripts/serve_vllm.sh
```

---

## 4. Phase 1 — sentence-level scoring

We maintain **four** benchmark parquets (see §4.1) and the same `scripts/score_sentences.py` pipeline runs against any of them via a uniform `--dataset-path` / `--dataset-name` pair.

```bash
# Step A — pull AggreFact (once). lytang/LLM-AggreFact is gated; run
# `huggingface-cli login` first.
python scripts/load_aggrefact.py --out data/aggrefact/aggrefact.parquet

# Step B — build the benchmark parquets.
python scripts/build_aggrefact_cnn.py          # -> data/aggrefact_cnn/aggrefact_cnn.parquet
python scripts/load_diversumm.py               # -> data/diversumm/diversumm.parquet
python scripts/build_aggrefact_other_multi.py  # -> data/aggrefact_other_multi/aggrefact_other_multi.parquet
python scripts/build_halueval.py               # -> data/halueval/halueval.parquet

# Step C — sentence-level scoring on the same machine:
ENDPOINT=http://<gpu-host>:8000/v1 \
MODEL_NAME=Qwen/Qwen3-8B \
python scripts/score_sentences.py --limit 20
# or
chmod +x ./scripts/exp.sh
./scripts/exp.sh <model_name>                  # llama-3.1-8b, olmo-3-7b, qwen-3-4b, qwen-3-8b, qwen-3-32b

# Step D - evaluation:
./scripts/eval.sh <model_name>                  # llama-3.1-8b, olmo-3-7b, qwen-3-4b, qwen-3-8b, qwen-3-32b
```

Four benchmarks:

| benchmark             | parquet                                                | rows | sent/summary (mean) | builder |
|-----------------------|--------------------------------------------------------|------|--------------------|--------------|
| **AggreFact-CNN**     | `data/aggrefact_cnn/aggrefact_cnn.parquet`             | 1017 | ~3.3 | `build_aggrefact_cnn.py` |
| **DiverSumm**         | `data/diversumm/diversumm.parquet`                     | 563  | 3–15 (origin-dep.) | `load_diversumm.py` |
| **AggreFact-other-≥2s** | `data/aggrefact_other_multi/aggrefact_other_multi.parquet` | 644  | 2.15 | `build_aggrefact_other_multi.py` |
| **HaluEval**          | `data/halueval/halueval.parquet`                       | 20000 derived rows from 10K source records | ~2.5 | `build_halueval.py` |

The original HaluEval summarization release contains 10K records. Each record includes both `right_summary` and `hallucinated_summary`; AMUSE expands that into two parquet rows per source record so sentence scoring and meta-eval can run on both candidate summaries. That derived parquet therefore contains 20K rows, but the original dataset does not separately list "10K faithful samples" as its own file.
