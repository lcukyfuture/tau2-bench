#!/usr/bin/env bash
set -euo pipefail

# Runs the full airline tau2 benchmark against a local OpenAI-compatible vLLM
# Gemma server. By default, both the agent and user simulator use Gemma.
#
# Common overrides:
#   SERVED_MODEL_NAME=gemma-local
#   BASE_URL=http://127.0.0.1:8003/v1
#   SAVE_TO=gemma_local_airline_full
#   MAX_CONCURRENCY=1
#   TIMEOUT=900
#   NUM_TASKS=10              # optional subset, omitted by default
#   TASK_IDS="1 2 3"          # optional explicit task IDs

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
NUM_TASKS="${NUM_TASKS:-}"
TASK_IDS="${TASK_IDS:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TIMEOUT="${TIMEOUT:-900}"
SAVE_TO="${SAVE_TO:-gemma_local_${DOMAIN}_full}"

cd "$REPO_ROOT"

args=(
  run
  --domain "$DOMAIN"
  --agent-llm "$AGENT_LLM"
  --agent-llm-args "$AGENT_LLM_ARGS"
  --user-llm "$USER_LLM"
  --user-llm-args "$USER_LLM_ARGS"
  --num-trials 1
  --max-concurrency "$MAX_CONCURRENCY"
  --timeout "$TIMEOUT"
  --verbose-logs
  --auto-resume
  --save-to "$SAVE_TO"
)

if [ -n "$NUM_TASKS" ]; then
  args+=(--num-tasks "$NUM_TASKS")
fi

if [ -n "$TASK_IDS" ]; then
  read -r -a task_ids_array <<< "$TASK_IDS"
  args+=(--task-ids "${task_ids_array[@]}")
fi

exec .venv/bin/tau2 "${args[@]}"
