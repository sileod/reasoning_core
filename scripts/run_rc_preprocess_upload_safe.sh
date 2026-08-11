#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMORY_MAX="${RC_MEMORY_MAX:-360G}"
PYTHON_BIN="${RC_PYTHON:-python3}"
LOG_DIR="${RC_LOG_DIR:-$ROOT/scripts/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/rc_preprocess_upload_$TS.log"

mkdir -p "$LOG_DIR"

cmd=("$PYTHON_BIN" "$ROOT/scripts/rc_preprocess_upload.py" "$@")

echo "log: $LOG"
echo "memory cap: $MEMORY_MAX"
echo "command: ${cmd[*]}"

if systemd-run --user --scope --quiet -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 true >/dev/null 2>&1; then
    runner=(systemd-run --user --scope --quiet -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 -p OOMPolicy=stop)
    "${runner[@]}" "${cmd[@]}" 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
else
    echo "systemd memory scope unavailable; falling back to ulimit -v" | tee "$LOG"
    case "$MEMORY_MAX" in
        *G) mem_kib=$(( ${MEMORY_MAX%G} * 1024 * 1024 )) ;;
        *M) mem_kib=$(( ${MEMORY_MAX%M} * 1024 )) ;;
        *) mem_kib="$MEMORY_MAX" ;;
    esac
    (
        ulimit -Sv "$mem_kib"
        ulimit -Hv "$mem_kib"
        exec "${cmd[@]}"
    ) 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
fi

echo "exit_status: $status" | tee -a "$LOG"
echo "log: $LOG"
exit "$status"
