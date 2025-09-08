#!/bin/bash

threads=$(python3 -c "import math, os; print(math.ceil(os.cpu_count() * 0.4))")

MEM_LIMIT="8G"

# --- Setup ---
# Directory for workers to report their status
STATUS_DIR="/dev/shm/gen_status_$$" # Using shared memory for performance
trap 'rm -rf "$STATUS_DIR"' EXIT # Cleanup on exit
mkdir -p "$STATUS_DIR"

echo "- Starting $threads persistent generation workers in the background..."
echo "- Workers will run until all examples are generated."
echo "- Dashboard will appear below. Press Ctrl-C to stop."

# --- Execution ---
# Run one persistent worker per thread. The python script itself contains the
# loop that continues until the --num_examples target is met.
# All arguments ("$@") are passed directly to the python workers.
seq "$threads" | parallel \
  -j"$threads" \
  --joblog generation.log \
  --line-buffer \
  systemd-run --user --scope -p MemoryMax="$MEM_LIMIT" \
  python generation_worker.py --id {} --status_dir "$STATUS_DIR" "$@" &

PARALLEL_PID=$!

# --- Monitoring ---
# This loop runs as long as the parallel command is alive. 'parallel' will
# exit only after all the python worker processes it launched have finished.
while ps -p $PARALLEL_PID > /dev/null; do
    clear
    echo "--- Generation Dashboard (PID: $PARALLEL_PID) --- ($(date +%H:%M:%S))"
    # Read all status files, sort them, and print them.
    grep . "$STATUS_DIR"/* 2>/dev/null | sort -V || echo "Waiting for workers to start..."
    sleep 1
done

echo "--- All workers finished. Parallel process complete. ---"
