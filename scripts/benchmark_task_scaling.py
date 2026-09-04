#!/usr/bin/env python3
"""Benchmark task generation across difficulty levels in isolated processes.

Examples:
  python scripts/benchmark_task_scaling.py --levels 0-6 --samples 3
  python scripts/benchmark_task_scaling.py --tasks table_qa planning --levels 0,3,6
  python scripts/benchmark_task_scaling.py --levels 6 --output-json level6.json
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue as queue_module
import statistics
import sys
import time
from datetime import datetime, timezone

from tabulate import tabulate

from reasoning_core import DEV_DATASETS, get_task, list_tasks


def parse_levels(spec):
    levels = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            if end < start:
                raise ValueError(f"invalid level range: {part}")
            levels.update(range(start, end + 1))
        else:
            levels.add(int(part))
    if not levels or min(levels) < 0:
        raise ValueError("levels must be non-negative")
    return sorted(levels)


def percentile(values, q):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def benchmark_cell(task_name, level, samples, max_tokens):
    task = get_task(task_name)
    timings, generation_times, prompt_tokens, answer_tokens = [], [], [], []
    deduplication_keys, prompts, errors = [], [], []
    valid_scores = 0

    for _ in range(samples):
        started = time.perf_counter()
        try:
            example = task.generate_example(level=level, max_tokens=max_tokens)
            timings.append(time.perf_counter() - started)
            valid_scores += int(task.score_answer(example.answer, example) == 1)
            generation_times.append(float(example.metadata.get("_time", timings[-1])))
            prompt_tokens.append(int(example.metadata.get("_prompt_tokens", 0)))
            answer_tokens.append(int(example.metadata.get("_answer_tokens", 0)))
            deduplication_keys.append(example.metadata.get("_deduplication_key"))
            prompts.append(example.prompt)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            break

    successes = len(timings)
    status = (
        "supported" if successes == samples and valid_scores == samples
        else "partial" if successes
        else "error"
    )
    return {
        "task": task_name,
        "level": level,
        "status": status,
        "requested": samples,
        "successes": successes,
        "valid_scores": valid_scores,
        "wall_seconds_mean": statistics.fmean(timings) if timings else None,
        "wall_seconds_first": timings[0] if timings else None,
        "wall_seconds_warm_mean": (
            statistics.fmean(timings[1:]) if len(timings) > 1 else None
        ),
        "wall_seconds_p50": statistics.median(timings) if timings else None,
        "wall_seconds_p95": percentile(timings, 0.95),
        "generation_seconds_mean": statistics.fmean(generation_times) if generation_times else None,
        "prompt_tokens_mean": statistics.fmean(prompt_tokens) if prompt_tokens else None,
        "prompt_tokens_max": max(prompt_tokens, default=None),
        "answer_tokens_mean": statistics.fmean(answer_tokens) if answer_tokens else None,
        "unique_prompt_ratio": len(set(prompts)) / successes if successes else None,
        "unique_deduplication_ratio": (
            len(set(deduplication_keys)) / successes if successes else None
        ),
        "config": task.config.to_dict(),
        "error": errors[0] if errors else None,
    }


def _cell_worker(queue, args):
    try:
        queue.put(benchmark_cell(*args))
    except Exception as exc:
        queue.put({
            "task": args[0], "level": args[1], "status": "error",
            "requested": args[2], "successes": 0, "valid_scores": 0,
            "error": f"{type(exc).__name__}: {exc}",
        })


def bounded_cell(task_name, level, samples, max_tokens, timeout):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_cell_worker,
        args=(queue, (task_name, level, samples, max_tokens)),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "task": task_name, "level": level, "status": "timeout",
            "requested": samples, "successes": 0, "valid_scores": 0,
            "error": f"cell exceeded {timeout:g}s",
        }
    try:
        return queue.get(timeout=1)
    except queue_module.Empty:
        return {
            "task": task_name, "level": level, "status": "error",
            "requested": samples, "successes": 0, "valid_scores": 0,
            "error": f"worker exited with code {process.exitcode}",
        }


def write_csv(path, rows):
    columns = [
        "task", "level", "status", "requested", "successes", "valid_scores",
        "wall_seconds_mean", "wall_seconds_first", "wall_seconds_warm_mean",
        "wall_seconds_p50", "wall_seconds_p95",
        "generation_seconds_mean", "prompt_tokens_mean", "prompt_tokens_max",
        "answer_tokens_mean", "unique_prompt_ratio", "unique_deduplication_ratio",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def display(rows):
    def number(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    table = [[
        row["task"], row["level"], row["status"],
        f"{row['successes']}/{row['requested']}",
        number(row.get("wall_seconds_mean")),
        number(row.get("wall_seconds_first")),
        number(row.get("wall_seconds_warm_mean")),
        number(row.get("wall_seconds_p95")),
        number(row.get("prompt_tokens_mean"), 1),
        number(row.get("unique_deduplication_ratio"), 2),
        row.get("error") or "",
    ] for row in rows]
    print(tabulate(table, headers=[
        "task", "level", "status", "ok", "mean_s", "first_s", "warm_s", "p95_s",
        "prompt_tok", "unique", "error",
    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", help="Task names; default: all core tasks")
    parser.add_argument("--levels", default="0-6", help="Comma-separated levels/ranges")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--cell-timeout", type=float, default=180)
    parser.add_argument("--include-dev", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--fail-on-unsupported", action="store_true")
    args = parser.parse_args()

    if args.samples < 1 or args.cell_timeout <= 0:
        parser.error("--samples and --cell-timeout must be positive")
    try:
        levels = parse_levels(args.levels)
    except ValueError as exc:
        parser.error(str(exc))

    available = list_tasks() + (list(DEV_DATASETS) if args.include_dev else [])
    tasks = args.tasks or available
    unknown = sorted(set(tasks) - set(available))
    if unknown:
        parser.error(f"unknown or excluded tasks: {', '.join(unknown)}")

    rows = []
    for task_name in tasks:
        for level in levels:
            print(f"benchmarking {task_name} level={level}", file=sys.stderr, flush=True)
            rows.append(bounded_cell(
                task_name, level, args.samples, args.max_tokens, args.cell_timeout
            ))

    display(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "levels": levels,
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "cell_timeout": args.cell_timeout,
        "rows": rows,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.fail_on_unsupported and any(row["status"] != "supported" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
