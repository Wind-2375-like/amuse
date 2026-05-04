# Add model name parameter
MODEL_NAME=$1

# Usage: ./scripts/eval.sh <model_name>
# chmod +x ./scripts/eval.sh first

# Mapping
# qwen-3-4b -> Qwen_Qwen3-4B
# qwen-3-8b -> Qwen_Qwen3-8B
# olmo-3-7b -> allenai_Olmo-3-7B
# llama-3.1-8b -> meta-llama_Llama-3.1-8B-Instruct

if [ "$MODEL_NAME" == "qwen-3-4b" ]; then
  MODEL_NAME="Qwen_Qwen3-4B"
elif [ "$MODEL_NAME" == "qwen-3-8b" ]; then
  MODEL_NAME="Qwen_Qwen3-8B"
elif [ "$MODEL_NAME" == "olmo-3-7b" ]; then
  MODEL_NAME="allenai_Olmo-3-7B-Instruct"
elif [ "$MODEL_NAME" == "llama-3.1-8b" ]; then
  MODEL_NAME="meta-llama_Llama-3.1-8B-Instruct"
else
  echo "Unknown model name: $MODEL_NAME"
  exit 1
fi

python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.$MODEL_NAME.aggrefact_cnn.parquet \
  --dataset data/aggrefact_cnn/aggrefact_cnn.parquet

python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.$MODEL_NAME.diversumm.parquet \
  --dataset data/diversumm/diversumm.parquet

python -m eval.run_meta_eval \
  --sentence-scores results/sentence_scores.$MODEL_NAME.aggrefact_other_multi.parquet \
  --dataset data/aggrefact_other_multi/aggrefact_other_multi.parquet