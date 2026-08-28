#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

: "${TASK_SEARCH_MODEL:?Set TASK_SEARCH_MODEL to an OpenCode model reference}"

task_search_seed=${TASK_SEARCH_SEED:-20260828}
task_search_queue=${TASK_SEARCH_QUEUE:-weekend_p0}

exec python -m reasoning_core.task_search run \
  reasoning_core/task_search/wave0.yaml \
  --model "$TASK_SEARCH_MODEL" \
  --queue "$task_search_queue" \
  --jobs 1 \
  --seed "$task_search_seed" \
  "$@"
