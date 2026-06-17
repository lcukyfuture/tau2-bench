#!/usr/bin/env bash
set -euo pipefail

SECRET_ENV_FILE="${SECRET_ENV_FILE:-.secrets/deepseek.env}"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$SECRET_ENV_FILE" ]; then
  set -a
  source "$SECRET_ENV_FILE"
  set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY must be set or provided in $SECRET_ENV_FILE." >&2
  exit 1
fi

TASKS_FILE="${TASKS_FILE:-data/tau2_noisy/domains/airline/redundant_information_v4_strict_all_tasks/tasks_redundant_information_v4_strict_all_tasks.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8003/v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-local}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TIMEOUT="${TIMEOUT:-900}"
SEED="${SEED:-300}"

USER_LLM="${USER_LLM:-openai/deepseek-v4-pro}"
USER_LLM_ARGS="${USER_LLM_ARGS:-{\"api_base\":\"https://api.deepseek.com\",\"temperature\":0.0,\"extra_body\":{\"thinking\":{\"type\":\"disabled\"}}}}"
INPUT_RECOVERY="${INPUT_RECOVERY:-0}"
INPUT_RECOVERY_LLM="${INPUT_RECOVERY_LLM:-$USER_LLM}"
INPUT_RECOVERY_LLM_ARGS="${INPUT_RECOVERY_LLM_ARGS:-$USER_LLM_ARGS}"

CLEAN_SAVE_TO="${CLEAN_SAVE_TO:-gemma_agent_deepseek_v4_pro_user_airline_clean_full_run1}"
NOISY_SAVE_TO="${NOISY_SAVE_TO:-gemma_agent_deepseek_v4_pro_user_airline_v4_strict_all_tasks_full_run1}"

RECOVERY_ARGS=()
if [ "$INPUT_RECOVERY" = "1" ]; then
  RECOVERY_ARGS+=(--input-recovery)
  RECOVERY_ARGS+=(--input-recovery-llm "$INPUT_RECOVERY_LLM")
  RECOVERY_ARGS+=(--input-recovery-llm-args "$INPUT_RECOVERY_LLM_ARGS")
fi

DOMAIN=airline \
SAVE_TO="$CLEAN_SAVE_TO" \
MAX_CONCURRENCY="$MAX_CONCURRENCY" \
TIMEOUT="$TIMEOUT" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
BASE_URL="$BASE_URL" \
USER_LLM="$USER_LLM" \
USER_LLM_ARGS="$USER_LLM_ARGS" \
scripts/run_gemma_tau2_airline_full.sh

.venv/bin/python scripts/noise/run_gemma_tau2_airline_noisy_file.py \
  --tasks-file "$TASKS_FILE" \
  --save-to "$NOISY_SAVE_TO" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --base-url "$BASE_URL" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$USER_LLM_ARGS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --seed "$SEED" \
  "${RECOVERY_ARGS[@]}"
