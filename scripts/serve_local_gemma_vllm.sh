#!/usr/bin/env bash
set -euo pipefail

# Starts a local OpenAI-compatible vLLM server for Gemma.
# Override these environment variables as needed:
#   GEMMA_MODEL_PATH=/path/to/model
#   SERVED_MODEL_NAME=gemma-local
#   GPU_IDS=2,3
#   PORT=8003
#   TP=2
#   VLLM_TOOL_CALL_PARSER=gemma4
#   VLLM_CHAT_TEMPLATE=/path/to/chat_template.jinja

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-/lp-dev/users/lingfeng/AgentFlow_noise/llm_models/google/gemma-4-31B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-local}"
GPU_IDS="${GPU_IDS:-2,3}"
PORT="${PORT:-8003}"
TP="${TP:-2}"
HOST="${HOST:-0.0.0.0}"
VLLM_BIN="${VLLM_BIN:-/home/lingfeng/lp-dev/AgentFlow/.venv/bin/vllm}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-model-len 32768 --max-num-batched-tokens 4096 --enforce-eager --generation-config vllm --chat-template-content-format openai}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-gemma4}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"

if [ ! -x "$VLLM_BIN" ]; then
  echo "vLLM binary not found or not executable: $VLLM_BIN" >&2
  exit 1
fi

if [ ! -e "$GEMMA_MODEL_PATH" ]; then
  echo "Gemma model path does not exist: $GEMMA_MODEL_PATH" >&2
  exit 1
fi

tool_args=()
if [ -n "$VLLM_TOOL_CALL_PARSER" ]; then
  tool_args+=(--enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_CALL_PARSER")
  if [ -n "$VLLM_CHAT_TEMPLATE" ]; then
    tool_args+=(--chat-template "$VLLM_CHAT_TEMPLATE")
  fi
else
  echo "Warning: VLLM_TOOL_CALL_PARSER is not set; tau2 tool calls may not work." >&2
fi

echo "Starting vLLM"
echo "  model: $GEMMA_MODEL_PATH"
echo "  served name: $SERVED_MODEL_NAME"
echo "  gpu ids: $GPU_IDS"
echo "  port: $PORT"
echo "  tensor parallel: $TP"
echo "  flashinfer sampler: $VLLM_USE_FLASHINFER_SAMPLER"

export VLLM_USE_FLASHINFER_SAMPLER

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_IDS" exec "$VLLM_BIN" serve "$GEMMA_MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${tool_args[@]}" \
  $VLLM_EXTRA_ARGS
