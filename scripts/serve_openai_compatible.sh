#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/local-awq}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "[serve] starting OpenAI-compatible server for ${MODEL_PATH}"
echo "[serve] install vllm first if needed: pip install vllm"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}"

