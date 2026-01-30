#!/bin/bash

threads=$(python3 -c "import math, os; print(math.ceil(os.cpu_count() * 0.5))")
#threads=1

MEM_LIMIT="12G"

STATUS_DIR="/dev/shm/gen_status_$$"
trap 'rm -rf "$STATUS_DIR"' EXIT
mkdir -p "$STATUS_DIR"

# 1. Capture Start Time
start_ts=$(date +%s)
echo "- Starting at: $(date)"
echo "- Starting $threads workers..."

seq "$threads" | parallel \
  -j"$threads" \
  --joblog generation.log \
  --line-buffer \
  systemd-run --user --scope -p MemoryMax="$MEM_LIMIT" \
  python generation_worker.py --id {} --status_dir "$STATUS_DIR" "$@" &

PARALLEL_PID=$!

while ps -p $PARALLEL_PID > /dev/null; do
    clear
    # 2. Calculate dynamic elapsed time for dashboard
    curr_ts=$(date +%s)
    elapsed=$(( curr_ts - start_ts ))
    curr_readable=$(date "+%H:%M:%S") 
    
    # Count non-zero exit codes (Col 7) in joblog, skipping header
    errs=$(awk 'NR>1 && $7!=0' generation.log 2>/dev/null | wc -l)

    echo "--- Dashboard (PID: $PARALLEL_PID) | Time: ${curr_readable} | Elapsed: ${elapsed}s | Errors: ${errs} ---"
    grep . "$STATUS_DIR"/* 2>/dev/null | sort -V || echo "Waiting for workers..."
    sleep 1
done

# 3. Final Stats
end_ts=$(date +%s)
total_time=$(( end_ts - start_ts ))
# accurate line count of all jsonl files in default output dir

echo "--- Finished. Duration: ${total_time}s.  ---"
