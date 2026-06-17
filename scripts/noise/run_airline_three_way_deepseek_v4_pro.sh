#!/usr/bin/env bash
set -euo pipefail

# Run three airline experiment variants with Gemma agents and DeepSeek V4 Pro users:
# 1. Clean airline on port 8003, 3 trials.
# 2. Noisy airline v4 strict all-tasks on port 8003, 3 trials, after clean completes.
# 3. Noisy airline v4 strict all-tasks with input recovery on port 8005, 3 trials.
#
# By default, the 8003 branch and 8005 recovery branch run in parallel. Within the
# 8003 branch, clean always finishes before noisy starts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

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

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_RECOVERY_PARALLEL="${RUN_RECOVERY_PARALLEL:-1}"

TASKS_FILE="${TASKS_FILE:-data/tau2_noisy/domains/airline/redundant_information_v4_strict_all_tasks/tasks_redundant_information_v4_strict_all_tasks.json}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-local}"
BASE_URL_8003="${BASE_URL_8003:-http://127.0.0.1:8003/v1}"
BASE_URL_8005="${BASE_URL_8005:-http://127.0.0.1:8005/v1}"

NUM_TRIALS="${NUM_TRIALS:-3}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TIMEOUT="${TIMEOUT:-900}"
SEED="${SEED:-300}"

USER_LLM="${USER_LLM:-openai/deepseek-v4-pro}"
USER_LLM_ARGS="${USER_LLM_ARGS:-{\"api_base\":\"https://api.deepseek.com\",\"temperature\":0.0,\"extra_body\":{\"thinking\":{\"type\":\"disabled\"}}}}"

AGENT_LLM="openai/$SERVED_MODEL_NAME"
AGENT_LLM_ARGS_8003="{\"api_base\":\"$BASE_URL_8003\",\"api_key\":\"EMPTY\",\"temperature\":0.0}"
AGENT_LLM_ARGS_8005="{\"api_base\":\"$BASE_URL_8005\",\"api_key\":\"EMPTY\",\"temperature\":0.0}"

CLEAN_SAVE_TO="${CLEAN_SAVE_TO:-gemma_agent_deepseek_v4_pro_user_airline_clean_3trials_run8003_${RUN_STAMP}}"
NOISY_SAVE_TO="${NOISY_SAVE_TO:-gemma_agent_deepseek_v4_pro_user_airline_v4_strict_all_tasks_3trials_run8003_${RUN_STAMP}}"
RECOVERY_SAVE_TO="${RECOVERY_SAVE_TO:-gemma_agent_deepseek_v4_pro_user_airline_v4_strict_all_tasks_input_recovery_3trials_run8005_${RUN_STAMP}}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/airline_three_way_${RUN_STAMP}.log"
BRANCH_8003_LOG="$LOG_DIR/airline_three_way_8003_${RUN_STAMP}.log"
BRANCH_8005_LOG="$LOG_DIR/airline_three_way_8005_recovery_${RUN_STAMP}.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MAIN_LOG"
}

run_clean_8003() {
  log "Starting clean airline: save_to=$CLEAN_SAVE_TO"
  .venv/bin/tau2 run \
    --domain airline \
    --agent-llm "$AGENT_LLM" \
    --agent-llm-args "$AGENT_LLM_ARGS_8003" \
    --user-llm "$USER_LLM" \
    --user-llm-args "$USER_LLM_ARGS" \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --timeout "$TIMEOUT" \
    --seed "$SEED" \
    --verbose-logs \
    --auto-resume \
    --save-to "$CLEAN_SAVE_TO"
  log "Finished clean airline: save_to=$CLEAN_SAVE_TO"
}

run_noisy_8003() {
  log "Starting noisy airline v4 strict all-tasks on 8003: save_to=$NOISY_SAVE_TO"
  .venv/bin/python scripts/noise/run_gemma_tau2_airline_noisy_file.py \
    --tasks-file "$TASKS_FILE" \
    --save-to "$NOISY_SAVE_TO" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --base-url "$BASE_URL_8003" \
    --user-llm "$USER_LLM" \
    --user-llm-args "$USER_LLM_ARGS" \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"
  log "Finished noisy airline v4 strict all-tasks on 8003: save_to=$NOISY_SAVE_TO"
}

run_recovery_8005() {
  log "Starting noisy airline v4 strict all-tasks with recovery on 8005: save_to=$RECOVERY_SAVE_TO"
  .venv/bin/python scripts/noise/run_gemma_tau2_airline_noisy_file.py \
    --tasks-file "$TASKS_FILE" \
    --save-to "$RECOVERY_SAVE_TO" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --base-url "$BASE_URL_8005" \
    --user-llm "$USER_LLM" \
    --user-llm-args "$USER_LLM_ARGS" \
    --input-recovery \
    --input-recovery-llm "$USER_LLM" \
    --input-recovery-llm-args "$USER_LLM_ARGS" \
    --num-trials "$NUM_TRIALS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"
  log "Finished noisy airline v4 strict all-tasks with recovery on 8005: save_to=$RECOVERY_SAVE_TO"
}

run_8003_branch() {
  run_clean_8003
  run_noisy_8003
}

log "Run stamp: $RUN_STAMP"
log "8003 clean save_to: $CLEAN_SAVE_TO"
log "8003 noisy save_to: $NOISY_SAVE_TO"
log "8005 recovery save_to: $RECOVERY_SAVE_TO"
log "num_trials=$NUM_TRIALS max_concurrency=$MAX_CONCURRENCY timeout=$TIMEOUT seed=$SEED"

if [ "$RUN_RECOVERY_PARALLEL" = "1" ]; then
  log "Launching 8003 branch and 8005 recovery branch in parallel"
  run_8003_branch >"$BRANCH_8003_LOG" 2>&1 &
  pid_8003=$!
  run_recovery_8005 >"$BRANCH_8005_LOG" 2>&1 &
  pid_8005=$!

  status=0
  if ! wait "$pid_8003"; then
    log "8003 branch failed; see $BRANCH_8003_LOG"
    status=1
  fi
  if ! wait "$pid_8005"; then
    log "8005 recovery branch failed; see $BRANCH_8005_LOG"
    status=1
  fi
  exit "$status"
else
  log "Launching all branches sequentially"
  run_8003_branch >"$BRANCH_8003_LOG" 2>&1
  run_recovery_8005 >"$BRANCH_8005_LOG" 2>&1
fi

log "All requested airline runs completed"
