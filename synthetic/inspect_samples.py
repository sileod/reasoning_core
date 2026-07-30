#!/usr/bin/env python3
"""
inspect_samples.py

Print a human-readable sample of generated rows, any time, from anything
accumulated so far in generated_data/llm_tasks/*.jsonl -- not just right
after a run. Use this to actually read the code the model wrote before
trusting a large batch of it.

    python synthetic/inspect_samples.py                        # 3 random rows per task
    python synthetic/inspect_samples.py --task code_runnability --n 8
    python synthetic/inspect_samples.py --task code_execution --level 2 --n 5
    python synthetic/inspect_samples.py --task code_execution --seed 1  # reproducible sample
    python synthetic/inspect_samples.py --check                 # also re-run each shown row through
                                                                   # the task's own real scorer
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reasoning_core as _rc  # noqa: E402  (import side-effect below prints its provenance once)
from reasoning_core import get_task  # noqa: E402
from package_for_hf import REAL_TASK_NAME, TOOLKIT_TASKS, revalidate  # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_row(row, idx, total, task=None, check=False):
    print(f"\n--- {row['task']}  sample {idx}/{total}  (level {row['level']}) ---")
    print("PROMPT:")
    print(row["prompt"])
    print("ANSWER:")
    print(row["answer"])
    if row.get("call_id") is not None:
        print(f"(call_id={row['call_id']}, source={row.get('source')})")
    if check and task is not None:
        ok, score = revalidate(task, row, row["task"])
        flag = "OK" if ok else f"FAILED (score={score!r})"
        print(f"RE-VALIDATION: {flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl-root", default="generated_data/llm_tasks")
    ap.add_argument("--task", default=None, help="Only this task. Default: every task found.")
    ap.add_argument("--n", type=int, default=3, help="Rows to show per task.")
    ap.add_argument("--level", type=int, default=None, help="Only rows at this level.")
    ap.add_argument("--seed", type=int, default=None, help="Fix the random sample for reproducibility.")
    ap.add_argument("--all", action="store_true", help="Show every matching row instead of sampling --n.")
    ap.add_argument("--check", action="store_true",
                     help="Re-run each shown row through the task's own real scorer (same check "
                          "package_for_hf.py does), so you get a pass/fail signal alongside the "
                          "visual read, not just eyeballing it.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    jsonl_root = Path(args.jsonl_root)
    files = sorted(jsonl_root.glob("*.jsonl"))
    if args.task:
        files = [f for f in files if f.stem == args.task]
    if not files:
        raise SystemExit(f"No matching .jsonl files under {jsonl_root} "
                          f"(--task {args.task!r} if that's set).")

    for path in files:
        public_name = path.stem
        rows = load_jsonl(path)
        if args.level is not None:
            rows = [r for r in rows if r["level"] == args.level]
        if not rows:
            print(f"\n{public_name}: no rows match (level filter: {args.level})")
            continue

        chosen = rows if args.all else random.sample(rows, min(args.n, len(rows)))
        task = None
        if args.check:
            real_name = REAL_TASK_NAME.get(public_name, public_name)
            task = get_task(real_name)

        print(f"\n{'=' * 78}\n{public_name}  ({len(rows)} total rows, showing {len(chosen)})\n{'=' * 78}")
        for i, row in enumerate(chosen, 1):
            print_row(row, i, len(chosen), task=task, check=args.check)


if __name__ == "__main__":
    main()