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

BATCH=0; threads=""
for i in "$@"; do
  [[ "$i" == "--batch" ]] && BATCH=1
  [[ "$prev" == "--threads" ]] && threads="$i"
  prev="$i"
done

# Default: 45% of CPUs, or use --threads override
[[ -z "$threads" ]] && threads=$(python3 -c "import math, os; print(math.ceil(os.cpu_count() * 0.45))")


STATUS_DIR="/dev/shm/gen_status_$$"
trap 'rm -rf "$STATUS_DIR" "$HF_HOME" "$NUMBA_CACHE_DIR"' EXIT
mkdir -p "$STATUS_DIR"

start_ts=$(date +%s)
echo "- Starting at: $(date)"
echo "- Starting $threads workers..."

seq "$threads" | parallel \
  -j"$threads" \
  --joblog generation.log \
  --line-buffer \
  'python generation_worker.py --id {} --status_dir '"$STATUS_DIR"' '"$@"'' &

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
