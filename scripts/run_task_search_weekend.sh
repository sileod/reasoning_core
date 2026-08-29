#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

: "${TASK_SEARCH_MODEL:?Set TASK_SEARCH_MODEL to an OpenCode model reference}"

task_search_seed=${TASK_SEARCH_SEED:-20260828}
task_search_queue=${TASK_SEARCH_QUEUE:-weekend_p0}
task_search_plan=${TASK_SEARCH_PLAN:-reasoning_core/task_search/wave0.yaml}
task_search_adapter=${TASK_SEARCH_ADAPTER:-direct}
task_search_harness=${TASK_SEARCH_HARNESS:-opencode}

if [[ -n ${TASK_SEARCH_KEY_FILE:-} ]]; then
  : "${TASK_SEARCH_KEY_ENV:?Set TASK_SEARCH_KEY_ENV with TASK_SEARCH_KEY_FILE}"
  printf -v "$TASK_SEARCH_KEY_ENV" '%s' "$(< "$TASK_SEARCH_KEY_FILE")"
  export "$TASK_SEARCH_KEY_ENV"
fi

adapter_args=(--adapter "$task_search_adapter")
if [[ $task_search_adapter == harness-link ]]; then
  : "${TASK_SEARCH_PROVIDER:?Set TASK_SEARCH_PROVIDER for Harness Link}"
  adapter_args+=(--provider "$TASK_SEARCH_PROVIDER")
fi
if [[ -n ${TASK_SEARCH_KEY_ENV:-} ]]; then
  adapter_args+=(--credential-env "$TASK_SEARCH_KEY_ENV")
fi

exec python -m reasoning_core.task_search run \
  "$task_search_plan" \
  --model "$TASK_SEARCH_MODEL" \
  --harness "$task_search_harness" \
  "${adapter_args[@]}" \
  --queue "$task_search_queue" \
  --jobs 1 \
  --seed "$task_search_seed" \
  --resource-limits required \
  "$@"
