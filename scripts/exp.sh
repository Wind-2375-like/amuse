python scripts/score_sentences.py \
    --dataset-path data/aggrefact_cnn/aggrefact_cnn.parquet \
    --dataset-name aggrefact_cnn \
    --model Qwen/Qwen3-4B

python scripts/score_sentences.py \
    --dataset-path data/diversumm/diversumm.parquet \
    --dataset-name diversumm \
    --model Qwen/Qwen3-4B

python scripts/score_sentences.py \
    --dataset-path data/aggrefact_other_multi/aggrefact_other_multi.parquet \
    --dataset-name aggrefact_other_multi \
    --model Qwen/Qwen3-4B