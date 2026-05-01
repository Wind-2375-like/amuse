#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for sentence-faithfulness scoring.
#
# Defaults: Qwen/Qwen3-8B, bf16, port 8000, 8K context, 1 GPU.
# Override via env vars, e.g.:
#   MODEL=Qwen/Qwen3-1.7B PORT=8001 ./scripts/serve_vllm.sh
#
# Requires: vllm installed in the active env. Install with:
#   pip install -e '.[serve]'
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP="${TP:-1}"

echo "[serve_vllm] model=$MODEL port=$PORT max_len=$MAX_LEN tp=$TP dtype=$DTYPE"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code
