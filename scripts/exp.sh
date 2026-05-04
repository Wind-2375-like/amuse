# Add model name parameter
MODEL_NAME=$1

# Usage: ./scripts/exp.sh <model_name>
# chmod +x ./scripts/exp.sh first

# Mapping
# qwen-3-4b -> Qwen/Qwen3-4B
# qwen-3-8b -> Qwen/Qwen3-8B
# qwen-3-32b -> Qwen/Qwen3-32B
# olmo-3-7b -> allenai/Olmo-3-7B
# llama-3.1-8b -> meta-llama/Llama-3.1-8B-Instruct

if [ "$MODEL_NAME" == "qwen-3-4b" ]; then
  MODEL_NAME="Qwen/Qwen3-4B"
elif [ "$MODEL_NAME" == "qwen-3-8b" ]; then
  MODEL_NAME="Qwen/Qwen3-8B"
elif [ "$MODEL_NAME" == "qwen-3-32b" ]; then
  MODEL_NAME="Qwen/Qwen3-32B"
elif [ "$MODEL_NAME" == "olmo-3-7b" ]; then
  MODEL_NAME="allenai/Olmo-3-7B-Instruct"
elif [ "$MODEL_NAME" == "llama-3.1-8b" ]; then
  MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
else
  echo "Unknown model name: $MODEL_NAME"
  exit 1
fi

python scripts/score_sentences.py \
    --dataset-path data/aggrefact_cnn/aggrefact_cnn.parquet \
    --dataset-name aggrefact_cnn \
    --model $MODEL_NAME

python scripts/score_sentences.py \
    --dataset-path data/diversumm/diversumm.parquet \
    --dataset-name diversumm \
    --model $MODEL_NAME

python scripts/score_sentences.py \
    --dataset-path data/aggrefact_other_multi/aggrefact_other_multi.parquet \
    --dataset-name aggrefact_other_multi \
    --model $MODEL_NAME