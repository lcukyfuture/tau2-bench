#!/usr/bin/env bash
set -euo pipefail

# Runs a one-task tau2 smoke test against a local OpenAI-compatible vLLM server.
# Assumes the server model name matches SERVED_MODEL_NAME.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-local}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8003/v1}"
LOCAL_LLM="openai/$SERVED_MODEL_NAME"
LOCAL_LLM_ARGS="{\"api_base\":\"$BASE_URL\",\"api_key\":\"EMPTY\",\"temperature\":0.0}"
DOMAIN="${DOMAIN:-airline}"
AGENT_LLM="${AGENT_LLM:-$LOCAL_LLM}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-$LOCAL_LLM_ARGS}"
USER_LLM="${USER_LLM:-$LOCAL_LLM}"
USER_LLM_ARGS="${USER_LLM_ARGS:-$LOCAL_LLM_ARGS}"
NUM_TASKS="${NUM_TASKS:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TIMEOUT="${TIMEOUT:-600}"
SAVE_TO="${SAVE_TO:-gemma_local_${DOMAIN}_smoke}"

cd "$REPO_ROOT"

exec .venv/bin/tau2 run \
  --domain "$DOMAIN" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$USER_LLM_ARGS" \
  --num-trials 1 \
  --num-tasks "$NUM_TASKS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --verbose-logs \
  --save-to "$SAVE_TO"
