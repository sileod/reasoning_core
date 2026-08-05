#!/usr/bin/env python3
"""
run_experiment_grid.py

One-shot, resumable orchestrator for the full procedural-vs-LLM-synthetic
scaling experiment. For every (model size x data source x seed), trains on
all 7 tasks and measures influence via the existing, unmodified
task_influence.py --run-influence pipeline.

Runs are sequential and --foreground on purpose: this is a single shared
GPU, so only one training job runs at a time (matching how your first run
already worked, one task after another). Every run's exact config and
outcome is appended to a manifest as it happens, and every run is preceded
by a completeness check (same logic task_influence.py's own launcher uses,
kept in sync deliberately) -- so re-running this script after an interrupted
night is always safe: already-finished runs are skipped, nothing is redone,
nothing is silently duplicated.

Run this ON THE GPU MACHINE. Two steps:

    # 1. One-time, CPU-only: build the procedural cache for all 7 tasks.
    #    (Your LLM-synthetic cache already exists at --llm-synth-cache.)
    python run_experiment_grid.py --build-procedural --procedural-n 200

    # 2. The full grid. Add --dry-run first to see the exact plan/commands
    #    without launching anything.
    python run_experiment_grid.py --dry-run
    python run_experiment_grid.py

Useful variations:
    python run_experiment_grid.py --models HuggingFaceTB/SmolLM2-135M   # just one scale first
    python run_experiment_grid.py --seeds 43                            # just one seed first
    python run_experiment_grid.py --sources llm_synth                  # just one arm first
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The 7 tasks in final scope (code_consolidation, path_convergence dropped).
# These are real, ALREADY-EXISTING reasoning_core tasks on the procedural
# side (code_execution.py / code_analysis.py) -- no new generators needed.
TASKS = [
    "code_execution", "code_runnability", "code_analysis", "code_input_deduction",
    "temporal_reasoning", "type_inhabitation", "code_repair",
]

# SmolLM2's three real published sizes -- same family/tokenizer/pretraining
# recipe across all three, which matters for a scaling claim not to be
# confounded by switching model families partway up the ladder.
MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
]

# Per-model batch/grad-accum. EFFECTIVE batch (batch * grad_accum) is kept
# identical across all three scales (=8) so runs are comparable rather than
# "whatever happened to fit" -- 1.7B halves the per-step batch and doubles
# accumulation to leave more headroom against OOM on the shared GPU. Watch
# your first 1.7B run (nvidia-smi in another pane) and adjust here if needed.
MODEL_TRAIN_CONFIG = {
    "HuggingFaceTB/SmolLM2-135M": dict(batch=8, grad_accum=1),
    "HuggingFaceTB/SmolLM2-360M": dict(batch=8, grad_accum=1),
    "HuggingFaceTB/SmolLM2-1.7B": dict(batch=4, grad_accum=2),
}

DEFAULT_SOURCES = {
    "procedural": "task_diagnostics/cache/task_rows/procedural_all7",
    "llm_synth": "task_diagnostics/cache/task_rows/llm_combined_all7",
}

DEFAULT_SEEDS = [43, 44, 45]


def short_model_tag(model_id):
    return model_id.split("/")[-1].replace(".", "p")


def run_tag_for(source, model_id, seed):
    # Deliberately does NOT include the seed: task_influence.py's own
    # run_result_paths() always appends "_S{seed}" to whatever run_tag it's
    # given, so baking the seed in here too would double it in the filename
    # (confirmed against a real completed run's actual output filename).
    return f"{source.upper()}_{short_model_tag(model_id)}"


def build_procedural_cache(n_per_task, levels, workers, out_dir):
    out_dir = Path(out_dir)
    print(f"\n=== building procedural cache: {len(TASKS)} tasks x {len(levels)} levels x "
          f"n={n_per_task} -> {out_dir}\n"
          f"    (note: code_runnability requires an EVEN n -- it generates OK/error pairs "
          f"atomically, 2 at a time)")
    if n_per_task % 2:
        raise SystemExit(f"--procedural-n must be even (code_runnability requires it), got {n_per_task}")
    cmd = [
        sys.executable, "-m", "task_diagnostics.cache", "build",
        "--tasks", *TASKS,
        "--levels", *[str(l) for l in levels],
        "--n", str(n_per_task),
        "--workers", str(workers),
        "--out", str(out_dir),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def result_path_for(run_tag, seed, train_steps, mix_aux, main_data, from_scratch):
    """Mirrors task_influence.py's own run_result_paths()/runner_results_dir() exactly --
    kept in sync deliberately so this script's pre-check agrees with the real one."""
    mix = int(mix_aux * 100)
    init = "scratch" if from_scratch else "pretrained"
    return ROOT / "per_task_results" / f"influence_{run_tag}_S{seed}_T{train_steps}_M{mix}_{main_data}_{init}.json"


def is_run_complete(expected_path, task_names):
    """Mirrors task_influence.py's own completed_result() exactly: the file must exist
    AND actually contain every requested task, not just exist."""
    if not expected_path.exists():
        return False
    try:
        data = json.loads(expected_path.read_text())
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("tasks"), dict) and all(
        t in data["tasks"] for t in task_names
    )


def append_manifest(manifest_path, record):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_one(source, model_id, seed, cache_dir, args, manifest_path):
    run_tag = run_tag_for(source, model_id, seed)
    cfg = MODEL_TRAIN_CONFIG.get(model_id, dict(batch=args.batch, grad_accum=args.grad_accum))
    expected = result_path_for(run_tag, seed, args.train_steps, args.mix_aux, args.main_data, args.from_scratch)

    if is_run_complete(expected, TASKS) and not args.force_run:
        print(f"[skip] {run_tag}: already complete -> {expected.name}")
        append_manifest(manifest_path, dict(
            run_tag=run_tag, source=source, model=model_id, seed=seed,
            status="skipped_already_done", result=str(expected),
            ts=datetime.now(timezone.utc).isoformat(),
        ))
        return

    cmd = [
        sys.executable, "task_diagnostics/task_influence.py", "--run-influence",
        "--taskrow-cache", str(cache_dir),
        "--tasks", *TASKS,
        "--model", model_id,
        "--seed", str(seed),
        "--train-steps", str(args.train_steps),
        "--mix-aux", str(args.mix_aux),
        "--main-data", args.main_data,
        "--batch", str(cfg["batch"]),
        "--grad-accum", str(cfg["grad_accum"]),
        "--run-tag", run_tag,
        "--no-eval-flan",
        "--foreground",
    ]
    if args.from_scratch:
        cmd += ["--from-scratch", "1"]
    if args.force_run:
        cmd.append("--force-run")

    env = dict(os.environ)
    if not args.no_extra_evals:
        env["EVAL_MBPP"] = "1"
        env["EVAL_MMLU_MATH"] = "1"
        env["EVAL_MMLU_LOGIC"] = "1"

    print(f"\n{'=' * 78}\n[launch] {run_tag}  (source={source}, model={model_id}, seed={seed})\n{'=' * 78}")
    print(" ".join(cmd))

    if args.dry_run:
        append_manifest(manifest_path, dict(
            run_tag=run_tag, source=source, model=model_id, seed=seed,
            status="dry_run", cmd=cmd, ts=datetime.now(timezone.utc).isoformat(),
        ))
        return

    t0 = time.time()
    append_manifest(manifest_path, dict(
        run_tag=run_tag, source=source, model=model_id, seed=seed,
        status="started", ts=datetime.now(timezone.utc).isoformat(),
    ))
    try:
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        status = "completed"
    except subprocess.CalledProcessError as e:
        status = f"failed (exit {e.returncode})"
        print(f"!! {run_tag} FAILED: {e}")
    dt = time.time() - t0
    append_manifest(manifest_path, dict(
        run_tag=run_tag, source=source, model=model_id, seed=seed,
        status=status, elapsed_s=round(dt, 1), result=str(expected),
        ts=datetime.now(timezone.utc).isoformat(),
    ))
    print(f"[{status}] {run_tag} in {dt / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-procedural", action="store_true",
                     help="Just build the procedural cache (CPU-only) and exit -- run this once, "
                          "before the main grid.")
    ap.add_argument("--procedural-n", type=int, default=200)
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--procedural-workers", type=int, default=8)
    ap.add_argument("--procedural-out", default=DEFAULT_SOURCES["procedural"])
    ap.add_argument("--llm-synth-cache", default=DEFAULT_SOURCES["llm_synth"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--sources", nargs="+", default=["procedural", "llm_synth"],
                     choices=["procedural", "llm_synth"])
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--mix-aux", type=float, default=0.2)
    ap.add_argument("--main-data", default="dolci")
    ap.add_argument("--batch", type=int, default=8, help="Default batch for any model not listed "
                     "in MODEL_TRAIN_CONFIG.")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--no-extra-evals", action="store_true",
                     help="Skip setting EVAL_MBPP/EVAL_MMLU_MATH/EVAL_MMLU_LOGIC=1 (on by default).")
    ap.add_argument("--force-run", action="store_true", help="Rerun even if already complete.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the full plan and every exact command; launch nothing.")
    ap.add_argument("--manifest", default="task_influence_work/experiment_grid_manifest.jsonl")
    args = ap.parse_args()

    if args.build_procedural:
        build_procedural_cache(args.procedural_n, args.levels, args.procedural_workers, args.procedural_out)
        return

    manifest_path = ROOT / args.manifest
    cache_by_source = {"procedural": args.procedural_out, "llm_synth": args.llm_synth_cache}

    plan = [
        (source, model_id, seed)
        for model_id in args.models
        for source in args.sources
        for seed in args.seeds
    ]
    print(f"Plan: {len(args.models)} model(s) x {len(args.sources)} source(s) x {len(args.seeds)} "
          f"seed(s) = {len(plan)} runs, each covering {len(TASKS)} tasks sequentially.")
    print(f"Manifest: {manifest_path}")

    # Write the exact config used for this invocation once, up front --
    # reproducibility means someone else (or future-you) can see precisely
    # what was asked for, not just what happened to run.
    append_manifest(manifest_path, dict(
        event="grid_start", ts=datetime.now(timezone.utc).isoformat(),
        tasks=TASKS, models=args.models, seeds=args.seeds, sources=args.sources,
        train_steps=args.train_steps, mix_aux=args.mix_aux, main_data=args.main_data,
        from_scratch=args.from_scratch, model_train_config=MODEL_TRAIN_CONFIG,
        cache_by_source=cache_by_source,
    ))

    t_start = time.time()
    for i, (source, model_id, seed) in enumerate(plan, 1):
        print(f"\n### run {i}/{len(plan)} ###")
        run_one(source, model_id, seed, cache_by_source[source], args, manifest_path)

    print(f"\nAll {len(plan)} planned runs processed in {(time.time() - t_start) / 3600:.2f}h.")
    print(f"Manifest: {manifest_path}")
    print(f"Raw results: per_task_results/influence_*.json")
    print(f"Next: python synthetic/build_report.py")


if __name__ == "__main__":
    main()