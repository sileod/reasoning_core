#!/usr/bin/env python3
"""
generate_one_each.py

Zero-cost sanity check: generate ONE example from each of the 7 in-scope tasks
using the real, unmodified reasoning_core pipeline (get_task().generate_entry()).

No LLM calls happen here. This just confirms:
  - every task imports and constructs cleanly in this environment
  - generate_entry() succeeds
  - render_prompt() / score_answer() round-trip on the reference answer

Run:
    python synthetic/generate_one_each.py
    python synthetic/generate_one_each.py --db-path functions.db --libraries numpy sklearn pandas scipy requests networkx yaml
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

# Self-contained: works regardless of cwd or PYTHONPATH, as long as this file
# sits exactly one level below the reasoning-core repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_core import get_task
import reasoning_core as _rc
print(f"(using reasoning_core from: {_rc.__file__})")

# task_name -> get_task() key. code_iterations is this project's rename of the
# upstream `temporal_reasoning` task (see handoff, section 1) -- the rename
# hasn't landed in reasoning-core itself yet, so we alias it here.
TASKS = {
    "code_execution": "code_execution",
    "code_runnability": "code_runnability",
    "code_analysis": "code_analysis",
    "code_iterations": "temporal_reasoning",
    "code_repair": "code_repair",
    "type_inhabitation": "type_inhabitation",
    "code_input_deduction": "code_input_deduction",
}

# The two toolkit-based tasks need functions.db. Building it against the
# default DEFAULT_LIBRARIES list is fine even when most of those libraries
# aren't installed (missing imports are skipped silently) -- takes ~10s.
TOOLKIT_TASKS = {"type_inhabitation", "code_repair"}


def _short(text, n=280):
    text = str(text)
    return text if len(text) <= n else text[:n].rstrip() + " …"


def run_one(public_name, real_name, db_path, libraries):
    kwargs = {}
    if public_name in TOOLKIT_TASKS:
        kwargs["config"] = _make_toolkit_config(real_name, db_path, libraries)

    t0 = time.time()
    task = get_task(real_name, **kwargs)
    entry = task.generate_entry()
    prompt = task.render_prompt(entry.metadata)
    score = task.score_answer(entry.answer, entry)
    dt = time.time() - t0

    print(f"\n{'=' * 78}\n{public_name}  (get_task key: {real_name!r}, {dt:.2f}s)\n{'=' * 78}")
    print("--- prompt ---")
    print(_short(prompt, 600))
    print("--- answer ---")
    print(_short(entry.answer, 300))
    print(f"--- self-score of reference answer (should be 1 / 1.0): {score}")
    print(f"--- metadata keys: {sorted(entry.metadata.keys())}")
    return True


def _make_toolkit_config(real_name, db_path, libraries):
    from reasoning_core.tasks.code_reasoning import TypeInhabitationCfg, CodeRepairCfg

    cfg_cls = TypeInhabitationCfg if real_name == "type_inhabitation" else CodeRepairCfg
    return cfg_cls(db_path=db_path, libraries=tuple(libraries) if libraries else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, choices=list(TASKS),
                     help="Only run these tasks (default: all 7).")
    ap.add_argument("--db-path", default="functions.db",
                     help="Sqlite function DB for type_inhabitation/code_repair "
                          "(built automatically on first use if missing).")
    ap.add_argument("--libraries", nargs="*", default=None,
                     help="Restrict the functions.db scrape to these installed "
                          "libraries (faster). Default: reasoning_core's own "
                          "DEFAULT_LIBRARIES list (missing ones are skipped).")
    args = ap.parse_args()

    wanted = args.only or list(TASKS)
    failures = []
    for public_name in wanted:
        real_name = TASKS[public_name]
        try:
            run_one(public_name, real_name, args.db_path, args.libraries)
        except Exception:
            failures.append(public_name)
            print(f"\n{'=' * 78}\n{public_name}  -- FAILED\n{'=' * 78}")
            traceback.print_exc()

    print(f"\n\n{len(wanted) - len(failures)}/{len(wanted)} tasks generated an example successfully.")
    if failures:
        print("Failed:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()