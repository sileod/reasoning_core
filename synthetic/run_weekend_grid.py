#!/usr/bin/env bash
# run_weekend_grid.sh
#
# One script, two phases. Meant to be launched inside a DETACHED TMUX SESSION
# (see commands below) so it survives you closing your laptop / SSH dropping.
#
#   - Safe to detach from / reattach to at ANY time (tmux handles this, not
#     this script).
#   - Safe to interrupt DURING PHASE 2 and resume later by just re-running
#     this script: run_experiment_grid.py's own completeness check skips any
#     (model, source, seed) combo that already has a finished result file.
#   - PHASE 1 (procedural cache build) should be left to finish uninterrupted.
#     It's CPU-only and much shorter than phase 2 -- there's no equivalent
#     resume logic for a half-built cache.
#
# IMPORTANT pre-flight (read once, do manually, not automated here):
#   Your existing n=50 result files at per_task_results/influence_PROCEDURAL_*.json
#   will look "already complete" to this script's skip-check even though they
#   were trained on n=50 data, not n=200. Move them aside first:
#
#     cd /mnt/nfs_share_magnet2/rrajanah/reasoning-core
#     mkdir -p per_task_results/_archived_n50
#     mv per_task_results/influence_PROCEDURAL_*.json per_task_results/_archived_n50/ 2>/dev/null
#
#   Also recommended: wipe the old n=50 procedural cache so there's no
#   ambiguity about which cache_id gets loaded at n=200 (it's cheap to
#   regenerate, CPU-only):
#
#     rm -rf task_diagnostics/cache/task_rows/procedural_all7

set -uo pipefail   # NOT -e: a failed phase should still leave clear logs, not vanish silently

# ---- env (same as your manual setup) ----
eval "$(/mnt/nfs_share_magnet2/rrajanah/envs/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/nfs_share_magnet2/rrajanah/envs/rc_exp
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/mnt/nfs_share_magnet2/rrajanah/hf_cache"
export HF_DATASETS_CACHE="/mnt/nfs_share_magnet2/rrajanah/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_DOWNLOAD_TIMEOUT=60
cd /mnt/nfs_share_magnet2/rrajanah/reasoning-core

LOG_DIR="task_influence_work/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

# Which sources phase 2 runs. Edit this if you also want llm_synth in the
# same weekend run, e.g.:  SOURCES="procedural llm_synth"
SOURCES="procedural"

echo "############################################################"
echo "# PHASE 1/2 -- building procedural cache, n=200/level (CPU only)"
echo "# started: $(date)"
echo "############################################################"
python synthetic/run_experiment_grid.py \
    --build-procedural --procedural-n 200 --procedural-workers 8 \
    2>&1 | tee "$LOG_DIR/phase1_build_procedural_${STAMP}.log"

N_MANIFESTS=$(find task_diagnostics/cache/task_rows/procedural_all7 -name manifest.json | wc -l)
echo "procedural manifest.json files found: $N_MANIFESTS"
if [ "$N_MANIFESTS" -lt 1 ]; then
    echo "!! phase 1 did not produce a manifest.json -- stopping before touching the GPU."
    exit 1
fi

echo "############################################################"
echo "# PHASE 2/2 -- full training grid (sources: $SOURCES)"
echo "# started: $(date)"
echo "############################################################"
python synthetic/run_experiment_grid.py --sources $SOURCES \
    2>&1 | tee -a "$LOG_DIR/phase2_grid_${STAMP}.log"

echo "############################################################"
echo "# ALL DONE: $(date)"
echo "############################################################"