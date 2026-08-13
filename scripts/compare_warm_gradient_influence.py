#!/usr/bin/env python3
"""Exploratory gradient-influence comparison along the historical warmup trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from reasoning_core.training.battery import load_battery_manifest
from reasoning_core.training.data import StreamSpec, load_stream
from reasoning_core.training.evals import contrastive_mc_loss, load_eval_suite
from reasoning_core.training.gradient_influence import completion_loss


MODEL = "HuggingFaceTB/SmolLM2-360M"
REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
TASKS = (
    "arithmetics", "code_runnability", "game_best_move", "graph_successors",
    "lean_candidate_compilation", "logic_nli", "metamath_core_select",
    "metamath_entailment", "most_probable_evidence", "most_probable_outcome",
    "parsing_derivation", "regex_following", "regex_reasoning", "rewrite_system",
    "string_transduction", "table_qa",
)


def sketch_gradients(model, dimensions, seed, *, scale=1.0):
    """CountSketch model gradients without materializing hashes or a flat gradient."""

    if dimensions < 1 or dimensions & (dimensions - 1):
        raise ValueError("dimensions must be a positive power of two")
    device = next(model.parameters()).device
    result = torch.zeros(dimensions, dtype=torch.float32, device=device)
    offset, chunk_size = 0, 4_000_000
    mask = dimensions - 1
    for parameter in model.parameters():
        if parameter.grad is None:
            offset += parameter.numel()
            continue
        flat = parameter.grad.detach().view(-1)
        for start in range(0, flat.numel(), chunk_size):
            values = flat[start:start + chunk_size].float().mul(scale)
            index = torch.arange(
                offset + start, offset + start + values.numel(), device=device,
                dtype=torch.int64,
            )
            bucket_hash = index * 1_103_515_245 + 12_345 + seed * 97_531
            sign_hash = index * 214_013 + 2_531_011 + seed * 65_537
            buckets = torch.bitwise_and(bucket_hash, mask)
            signs = 1.0 - 2.0 * torch.bitwise_and(sign_hash >> 16, 1).float()
            result.scatter_add_(0, buckets, values * signs)
        offset += flat.numel()
    return result


def gradient_norm(model):
    return math.sqrt(sum(
        parameter.grad.detach().float().square().sum().item()
        for parameter in model.parameters() if parameter.grad is not None
    ))


def build_arc_sketch(model, tokenizer, legs, max_length, dimensions, seed):
    started = time.perf_counter()
    aggregate = torch.zeros(dimensions, dtype=torch.float32, device=model.device)
    leg_details = []
    for name, examples in legs.items():
        model.zero_grad(set_to_none=True)
        scored = tokens = 0
        for example in examples:
            result = contrastive_mc_loss(model, tokenizer, (example,), max_length)
            (result.loss * result.examples).backward()
            scored += result.examples
            tokens += result.tokens
        norm = gradient_norm(model)
        aggregate.add_(sketch_gradients(model, dimensions, seed, scale=1.0 / norm))
        leg_details.append({"name": name, "examples": scored, "tokens": tokens, "norm": norm})
    model.zero_grad(set_to_none=True)
    aggregate.div_(aggregate.norm())
    torch.cuda.synchronize()
    return aggregate, leg_details, time.perf_counter() - started


def score_tasks(model, tokenizer, task_rows, eval_sketch, args):
    records = {}
    for task in TASKS:
        rows = task_rows[task]
        cosines, accumulated = [], torch.zeros_like(eval_sketch)
        started = time.perf_counter()
        for batch in range(args.batches):
            model.zero_grad(set_to_none=True)
            loss = completion_loss(
                model, tokenizer, rows[batch * args.batch_size:(batch + 1) * args.batch_size],
                args.max_length,
            )
            loss.loss.backward()
            task_sketch = sketch_gradients(model, args.projection_dimensions, args.projection_seed)
            accumulated.add_(task_sketch)
            cosines.append(float(torch.dot(eval_sketch, task_sketch) / task_sketch.norm()))
        torch.cuda.synchronize()
        records[task] = {
            "cosine": float(torch.dot(eval_sketch, accumulated) / accumulated.norm()),
            "stderr": float(np.std(cosines, ddof=1) / math.sqrt(len(cosines))),
            "seconds": time.perf_counter() - started,
        }
        print(f"  {task:36s} {records[task]['cosine']:+.6f}", flush=True)
    model.zero_grad(set_to_none=True)
    return records


class WarmProbe(TrainerCallback):
    def __init__(self, tokenizer, legs, task_rows, truth, args):
        self.tokenizer, self.legs, self.task_rows = tokenizer, legs, task_rows
        self.truth, self.args, self.results = truth, args, {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step not in self.args.warm_steps or str(step) in self.results:
            return control
        print(f"\n[warm_{step}] building projected ARC gradient", flush=True)
        was_training = model.training
        model.eval()
        sketch, legs, cache_seconds = build_arc_sketch(
            model, self.tokenizer, self.legs, self.args.max_length,
            self.args.projection_dimensions, self.args.projection_seed,
        )
        print(f"[warm_{step}] scoring tasks", flush=True)
        records = score_tasks(model, self.tokenizer, self.task_rows, sketch, self.args)
        self.results[str(step)] = {
            "cache_seconds": cache_seconds,
            "legs": legs,
            "records": records,
            "metrics": metrics(records, self.truth),
        }
        write_output(self.results, self.truth, self.args)
        model.train(was_training)
        if step == max(self.args.warm_steps):
            control.should_training_stop = True
        return control


def metrics(records, truth):
    tasks = list(records)
    predicted = [records[task]["cosine"] for task in tasks]
    result = {}
    for target in ("true_margin", "true_nll_improvement"):
        actual = [truth[task][target] for task in tasks]
        rho = spearmanr(predicted, actual)
        pearson = pearsonr(predicted, actual)
        result[target] = {
            "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
            "pearson": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "pairwise": pairwise(predicted, actual),
        }
    result["mean_stderr"] = float(np.mean([records[task]["stderr"] for task in tasks]))
    result["mean_seconds_per_task"] = float(np.mean([records[task]["seconds"] for task in tasks]))
    return result


def pairwise(predicted, actual):
    agree = total = 0
    for left in range(len(actual)):
        for right in range(left + 1, len(actual)):
            if actual[left] == actual[right]:
                continue
            agree += np.sign(actual[left] - actual[right]) == np.sign(predicted[left] - predicted[right])
            total += 1
    return agree / total


def write_output(results, truth, args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "model": MODEL, "revision": REVISION, "main_source": args.main_source,
            "warm_steps": args.warm_steps, "max_length": args.max_length,
            "projection": "countsketch_v1", "projection_dimensions": args.projection_dimensions,
            "projection_seed": args.projection_seed, "task_batches": args.batches,
            "task_batch_size": args.batch_size,
        },
        "states": results,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["task", "true_margin", "true_nll_improvement"]
    fields += [f"warm_{step}" for step in args.warm_steps if str(step) in results]
    with output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            row = {"task": task, **truth[task]}
            row.update({f"warm_{step}": results[str(step)]["records"][task]["cosine"]
                        for step in args.warm_steps if str(step) in results})
            writer.writerow(row)


def load_inputs(tokenizer, args):
    history = json.loads(Path(args.historical_results).read_text())
    truth = {}
    for task in TASKS:
        row = history["tasks"][task]
        truth[task] = {
            "true_margin": np.mean([
                row["arc_easy_mc_cloze_margin_delta"],
                row["arc_challenge_mc_cloze_margin_delta"],
            ]).item(),
            "true_nll_improvement": -np.mean([
                row["arc_easy_delta"], row["arc_challenge_delta"],
            ]).item(),
        }
    battery = load_battery_manifest(args.battery, args.data_dir, args.max_length)
    legs = {
        leg.name: load_eval_suite(
            leg.path, tokenizer.eos_token, name=leg.name, limit=leg.limit,
        ).examples
        for leg in battery.legs if leg.name in {"arc_easy", "arc_challenge"}
    }
    task_rows = {}
    for task in TASKS:
        spec = StreamSpec(args.task_source, "influence_legacy_v1", task=task)
        task_rows[task] = list(load_stream(
            spec, tokenizer, args.max_length, max_tokens=args.max_length - 8,
        ).take(args.batches * args.batch_size))
    return truth, legs, task_rows


def main():
    args = parse_args()
    torch.manual_seed(43)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    truth, legs, task_rows = load_inputs(tokenizer, args)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    probe = WarmProbe(tokenizer, legs, task_rows, truth, args)
    dataset = load_stream(StreamSpec(
        args.main_source, "influence_auto_v1", cycle=True,
    ), tokenizer)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset, callbacks=[probe],
        args=SFTConfig(
            output_dir=args.trainer_output, max_steps=300,
            per_device_train_batch_size=4, gradient_accumulation_steps=2,
            learning_rate=1e-4, weight_decay=0.01, lr_scheduler_type="linear",
            optim="adamw_torch", max_grad_norm=1.0, max_length=args.max_length,
            completion_only_loss=True, packing=True, bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none", logging_steps=10, save_strategy="no", seed=43,
            disable_tqdm=True,
        ),
    )
    trainer.train()
    print(json.dumps({step: value["metrics"] for step, value in probe.results.items()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-source", default="data_cache/fwdolci_main.jsonl")
    parser.add_argument("--task-source", default="task_diagnostics/cache/task_rows/75c0b75b1001")
    parser.add_argument("--historical-results", default="per_task_results/influence_COLL-roster_ROSTER2_S43_S43_T300_M20_fwdolci_pretrained.json")
    parser.add_argument("--battery", default="reasoning_core/training/copyfree_battery_v3.json")
    parser.add_argument("--data-dir", default="data_cache")
    parser.add_argument("--output", default="task_influence_work/gradient_arc_warm_comparison_sm360_s43")
    parser.add_argument("--trainer-output", default="task_influence_work/warm_probe_trainer")
    parser.add_argument("--warm-steps", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dimensions", type=int, default=262_144)
    parser.add_argument("--projection-seed", type=int, default=43)
    return parser.parse_args()


if __name__ == "__main__":
    main()
