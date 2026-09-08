#!/usr/bin/env python3
"""Calibrate gradient alignment against the published 51-task influence ranking."""

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_core import get_task
from reasoning_core.evaluation.battery import load_battery_manifest
from reasoning_core.evaluation.training.data import format_row
from reasoning_core.evaluation.metrics import load_eval_suite
from reasoning_core.evaluation.gradient import (
    GradientCacheSpec,
    build_eval_gradient_cache,
    gradient_objective_id,
    load_gradient_cache,
    score_task,
)


BATCH_COUNTS = (1, 2, 4, 8, 16, 32)
TINY_MODEL = "sileod/microlm-ettin-swa-5m"
TINY_REVISION = "38714a366a73da97d8c17f554f0f428a79418cf1"


def main():
    args = parse_args()
    truth = load_truth(args.truth)
    task_rows = load_task_rows(args.task_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    battery = load_battery_manifest(args.battery, args.data_dir, args.max_length)
    selected, legs = set(args.legs), {}
    for leg in battery.legs:
        if leg.name in selected:
            if leg.kind != "mcq":
                raise ValueError(f"Gradient benchmark leg {leg.name!r} is not MCQ")
            legs[leg.name] = load_eval_suite(
                leg.path, tokenizer.eos_token, name=leg.name, limit=leg.limit,
            ).examples
    if selected - legs.keys():
        raise ValueError(f"Battery lacks legs: {', '.join(sorted(selected - legs.keys()))}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype,
    ).to(args.device)
    objective_id = gradient_objective_id(legs, args.max_length, args.temperature)
    spec = GradientCacheSpec(
        args.initialization_id, objective_id, args.max_length, args.cache_dtype,
    )
    cache_seconds = 0.0
    try:
        cache = load_gradient_cache(spec, device=args.device, cache_dir=args.cache_dir)
    except FileNotFoundError:
        started = time.perf_counter()
        cache = build_eval_gradient_cache(
            model, tokenizer, legs, spec, temperature=args.temperature,
            cache_dir=args.cache_dir, model_id=args.model, model_revision=args.revision,
        )
        _synchronize(args.device)
        cache_seconds = time.perf_counter() - started

    records = []
    timings = {count: [] for count in BATCH_COUNTS}
    stderrs = {count: [] for count in BATCH_COUNTS}
    for task_name, true_influence in truth.items():
        factory = batch_factory(
            task_name, tokenizer, task_rows.get(task_name), args.batch_size, args.level,
        )
        row = {"task": task_name, "true_influence": true_influence}
        for count in BATCH_COUNTS:
            _synchronize(args.device)
            started = time.perf_counter()
            estimate = score_task(
                model, tokenizer, factory, cache, batches=count,
                max_length=args.max_length, seed=args.seed,
            )
            _synchronize(args.device)
            timings[count].append(time.perf_counter() - started)
            stderrs[count].append(estimate.stderr_cosine)
            row[f"grad_{count}"] = estimate.aggregate_cosine
        records.append(row)
        print(f"{task_name}: grad_32={row['grad_32']:+.5f}", flush=True)

    metrics = {
        "cache_gpu_seconds": cache_seconds,
        "tasks": len(records),
        "by_batches": {
            str(count): calibration_metrics(records, count, timings[count], stderrs[count])
            for count in BATCH_COUNTS
        },
        "spec": {
            "model": args.model, "revision": args.revision,
            "initialization_id": args.initialization_id,
            "objective_id": objective_id, "seed": args.seed,
            "batch_size": args.batch_size,
        },
    }
    write_outputs(records, metrics, args.output)
    print(json.dumps(metrics, indent=2))


def batch_factory(task_name, tokenizer, stored_rows, batch_size, level):
    if stored_rows:
        rows = [_format(row, tokenizer) for row in stored_rows]

        def from_rows(seed):
            rng = random.Random(f"{task_name}:{seed}")
            if len(rows) < batch_size:
                raise RuntimeError(f"{task_name} has only {len(rows)} stored rows")
            return rng.sample(rows, batch_size)

        return from_rows

    def generate(_seed):
        task = get_task(task_name)
        task.config.set_level(level)
        examples = task.generate_balanced_batch(
            batch_size, workers=1, deduplication=True,
        )
        return [_format(example, tokenizer) for example in examples]

    return generate


def load_truth(path):
    text = Path(path).read_text()
    rows = re.findall(
        r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*\*\*([+-]?[\d.]+)%\*\*", text, re.M,
    )
    if not rows:
        raise RuntimeError(f"No historical influence rows found in {path}")
    return {task: float(value) for task, value in rows}


def load_task_rows(paths):
    merged = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, list):
            for row in payload:
                merged.setdefault(row["task"], []).append(row)
        else:
            for task, rows in payload.items():
                merged.setdefault(task, []).extend(rows)
    return merged


def calibration_metrics(records, count, timings, stderrs):
    true = pd.Series([row["true_influence"] for row in records])
    predicted = pd.Series([row[f"grad_{count}"] for row in records])
    return {
        "spearman_rho": float(true.corr(predicted, method="spearman")),
        "pairwise_ranking_accuracy": pairwise_accuracy(true, predicted),
        "top_5_recall": top_recall(true, predicted, 5),
        "top_10_recall": top_recall(true, predicted, 10),
        "mean_score_stderr": float(np.mean(stderrs)),
        "gpu_seconds_per_task": float(np.mean(timings)),
    }


def pairwise_accuracy(true, predicted):
    correct = total = 0
    for left in range(len(true)):
        for right in range(left + 1, len(true)):
            true_sign = np.sign(true[left] - true[right])
            predicted_sign = np.sign(predicted[left] - predicted[right])
            if true_sign:
                correct += true_sign == predicted_sign
                total += 1
    return correct / total if total else math.nan


def top_recall(true, predicted, k):
    k = min(k, len(true))
    return len(set(true.nlargest(k).index) & set(predicted.nlargest(k).index)) / k


def write_outputs(records, metrics, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task", "true_influence", *(f"grad_{n}" for n in BATCH_COUNTS)]
    with output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    output.with_suffix(".json").write_text(json.dumps({
        "records": records, "metrics": metrics,
    }, indent=2) + "\n")


def _format(row, tokenizer):
    if "completion" in row:
        return {"prompt": str(row["prompt"]), "completion": str(row["completion"])}
    return format_row(row, tokenizer.eos_token, "sft_qa_v1")


def _synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=TINY_MODEL, help="Warmed model/checkpoint")
    parser.add_argument("--revision")
    parser.add_argument("--initialization-id")
    parser.add_argument("--battery", default="reasoning_core/training/paper_battery.json")
    parser.add_argument("--data-dir", default="data_cache")
    parser.add_argument("--legs", nargs="+", default=["arc_easy", "arc_challenge"])
    parser.add_argument("--task-rows", nargs="*", default=[])
    parser.add_argument("--truth", default="docs/results/influence.md")
    parser.add_argument("--output", default="gradient_influence_validation")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--cache-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.model == TINY_MODEL and args.revision is None:
        args.revision = TINY_REVISION
    if args.initialization_id is None:
        if args.model != TINY_MODEL:
            parser.error("--initialization-id is required for a non-default warmed model")
        args.initialization_id = f"hf:{TINY_MODEL}@{args.revision}"
    return args


if __name__ == "__main__":
    main()
