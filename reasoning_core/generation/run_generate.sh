#!/bin/bash

# Thread controls
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# NFS protection - move caches to local temp
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="/tmp/hf_$$"
export NUMBA_CACHE_DIR="/tmp/numba_$$"
mkdir -p "$HF_HOME" "$NUMBA_CACHE_DIR" 2>/dev/null

# Split launcher options from worker options. These three are consumed HERE; everything else is
# forwarded. They used to be forwarded too, so `--threads 8` reached the worker's argparse as an
# unknown argument and killed every task -- the launcher's own documented option broke the run.
BATCH=0; threads=""; script_dir="."
worker_args=()
while (( $# )); do
  case "$1" in
    --batch)      BATCH=1; shift ;;
    --threads)    threads="$2"; shift 2 ;;
    --script_dir) script_dir="$2"; shift 2 ;;
    *)            worker_args+=("$1"); shift ;;
  esac
done
# Quote once here so the string can be embedded in the single command GNU parallel runs.
printf -v WORKER_ARGS '%q ' "${worker_args[@]}"

# Default: 45% of CPUs, or use --threads override
[[ -z "$threads" ]] && threads=$(python3 -c "import math, os; print(math.ceil(os.cpu_count() * 0.4))")


STATUS_DIR="/dev/shm/gen_status_$$"
trap 'rm -rf "$STATUS_DIR" "$HF_HOME" "$NUMBA_CACHE_DIR"' EXIT
mkdir -p "$STATUS_DIR"

# Launch the worker as a MODULE, not a file path. This script used to exec
# reasoning_core/generation_worker.py by path; when that module moved, the launcher kept
# "working" while doing nothing at all, because the file it found was a re-export stub with no
# __main__. A module launch fails loudly instead, and survives the worker being moved again.
python -c "import reasoning_core.generation.worker" 2>/dev/null || {
  echo "!!! cannot import reasoning_core.generation.worker -- run from the repo root or install the package" >&2
  exit 3
}

start_ts=$(date +%s)
echo "- Starting at: $(date)"
echo "- Starting $threads workers..."

MEM_LIMIT_KB=$((50*1024*1024))  # 50GB in KB
seq $((threads * 200)) | parallel \
  --workdir "$PWD" \
  -j"$threads" \
  --joblog "$script_dir/generation.log" \
  --line-buffer \
  'ulimit -v '"$MEM_LIMIT_KB"' 2>/dev/null; timeout --signal=KILL 1000 python -m reasoning_core.generation.worker --id {} --status_dir '"$STATUS_DIR"' --out_path "'"$script_dir"'/generated_data" '"$WORKER_ARGS"'' &

PARALLEL_PID=$!

if [[ -z "$OAR_JOB_ID" && "$BATCH" -eq 0 ]]; then
  while ps -p $PARALLEL_PID > /dev/null; do
    clear
    curr_ts=$(date +%s); elapsed=$(( curr_ts - start_ts ))
    errs=$(awk 'NR>1 && $7!=0' generation.log 2>/dev/null | wc -l)
    echo "--- Dashboard | Elapsed: ${elapsed}s | Errors: ${errs} ---"
    for f in "$STATUS_DIR"/*; do
      [ -f "$f" ] || continue
      line=$(cat "$f" 2>/dev/null) || continue
      ts=$(echo "$line" | grep -oP 'ts:\K[0-9]+' || echo "")
      if [ -n "$ts" ]; then
        task_elapsed=$(( curr_ts - ts ))
        # Remove ts:... suffix and append elapsed time
        clean_line=$(echo "$line" | sed 's/ | ts:[0-9]*//')
        echo "${clean_line} | Elapsed: ${task_elapsed}s"
      else
        echo "$line"
      fi
    done | sort -V || echo "Waiting..."
    if [ -f errors.log ]; then echo "--- Last Errors ---"; tail -3 errors.log; fi
    sleep 1
  done
else
  wait $PARALLEL_PID
fi

end_ts=$(date +%s)
echo "--- Finished. Duration: $((end_ts - start_ts))s. ---"
