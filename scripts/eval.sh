python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.Qwen_Qwen3-4B.aggrefact_cnn.parquet \
  --dataset data/aggrefact_cnn/aggrefact_cnn.parquet

python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.Qwen_Qwen3-4B.diversumm.parquet \
  --dataset data/diversumm/diversumm.parquet

python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.Qwen_Qwen3-4B.aggrefact_other_multi.parquet \
  --dataset data/aggrefact_other_multi/aggrefact_other_multi.parquet