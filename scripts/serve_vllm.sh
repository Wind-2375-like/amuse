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

# Workaround: vLLM 0.20.x on H100 hits a DeepGEMM warmup path even for bf16
# models (FP8 kernels not in use). Disable to avoid `deep_gemm` import error.
# Override by exporting VLLM_USE_DEEP_GEMM=1 if you actually want FP8.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

echo "[serve_vllm] model=$MODEL port=$PORT max_len=$MAX_LEN tp=$TP dtype=$DTYPE"
echo "[serve_vllm] VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code
