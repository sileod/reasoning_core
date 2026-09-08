import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
from datasets import load_dataset

from reasoning_core import score_answer
from reasoning_core.template import Entry


REWARD_VERSION = 2


@dataclass(frozen=True)
class FreeGenRewardSpec:
    """Small, explicit protocol for task-native free-generation reward."""

    mode: str = "instruct"
    n_eval: int = 25
    max_tokens: int = 256

    def __post_init__(self):
        if self.n_eval < 1 or self.max_tokens < 1:
            raise ValueError("n_eval and max_tokens must be positive")


def reward_id(rows, spec, max_length):
    payload = {
        "version": REWARD_VERSION,
        "spec": asdict(spec),
        "max_length": max_length,
        "rows": [_row_dict(row) for row in rows],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"task_reward@v{REWARD_VERSION}:{hashlib.sha256(encoded.encode()).hexdigest()[:12]}"


@torch.no_grad()
def free_gen_reward(model, tokenizer, rows, spec, max_length):
    """Greedily generate answers and score them with each task's native scorer."""

    rows = [_row_dict(row) for row in rows]
    if spec.mode not in ("", "all"):
        matching = [row for row in rows if (row.get("mode") or "") == spec.mode]
        if len(matching) >= 5:
            rows = matching

    was_training = model.training
    model.eval()
    scores = []
    try:
        for row in rows[:spec.n_eval]:
            prompt_ids = _token_ids(tokenizer, f"{row['prompt']}\n")
            answer_ids = _token_ids(tokenizer, f"{row['answer']}{tokenizer.eos_token}")
            if not answer_ids or len(prompt_ids) >= max_length:
                continue
            cap = min(len(answer_ids) + 8, spec.max_tokens, max_length - len(prompt_ids))
            inputs = torch.tensor([prompt_ids], device=next(model.parameters()).device)
            output = model.generate(
                inputs, max_new_tokens=cap, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            prediction = tokenizer.decode(
                output[0, len(prompt_ids):], skip_special_tokens=True,
            )
            score = _native_score(row, prediction)
            if score is not None:
                scores.append(score)
    finally:
        model.train(was_training)
    return {
        "reward": sum(scores) / len(scores) if scores else None,
        "reward_examples": len(scores),
    }


def load_intrinsic_eval_split(path, skip=100_000, max_groups=200, max_examples_per_group=8, max_scanned=50_000):
    buckets = defaultdict(list)
    for i, row in enumerate(load_dataset(path, split="train", streaming=True).skip(skip)):
        if i >= max_scanned or (len(buckets) >= max_groups and all(len(v) >= max_examples_per_group for v in buckets.values())):
            break
        metadata = _metadata(row)
        task = _task(row, metadata)
        level = row.get("level", metadata.get("_level", ""))
        if not (task and level != "" and row.get("prompt") and row.get("answer")):
            continue
        group = ".".join(_key(v) for v in (task, level))
        if group not in buckets and len(buckets) >= max_groups:
            continue
        if len(buckets[group]) < max_examples_per_group:
            metadata.setdefault("_task", row.get("task") or metadata.get("task") or task)
            metadata.setdefault("_level", level)
            buckets[group].append({"prompt": row["prompt"], "answer": row["answer"], "metadata": metadata})
    print(f"intrinsic eval buckets ({len(buckets)} kept): {sorted((k, len(v)) for k, v in buckets.items())}")
    return dict(buckets)


def log_intrinsic_task_rewards(model, tokenizer, splits, sink, global_step=None, max_steps=None, max_new_tokens=64):
    import wandb

    metrics, by_task = {}, defaultdict(list)
    for group, rows in splits.items():
        vals = [value for row in rows
                if (value := _reward(model, tokenizer, row, max_new_tokens)) is not None]
        if vals:
            task = group.rsplit(".", 1)[0]
            metrics[f"intrinsic_reward_tl/{group}"] = sum(vals) / len(vals)
            by_task[task].extend(vals)
    for task, vals in by_task.items():
        metrics[f"intrinsic_reward/{task}"] = sum(vals) / len(vals)
    metrics["intrinsic_eval/n"] = sum(map(len, by_task.values()))
    wandb.log(metrics, step=global_step) if global_step is not None else wandb.log(metrics)
    for key, value in metrics.items():
        if key.startswith("intrinsic_reward/"):
            wandb.run.summary[f"compare/{key.replace('/', '_')}"] = value
    sink.record(metrics, kind="intrinsic", global_step=global_step, max_steps=max_steps)


def _reward(model, tokenizer, row, max_new_tokens):
    prompt = f"Q: {row['prompt']}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return _native_score(row, pred)


def _native_score(row, prediction):
    if " ".join(str(prediction).split()) == " ".join(str(row["answer"]).split()):
        return 1.0
    metadata = _metadata(row)
    if row.get("task"):
        metadata.setdefault("_task", row["task"])
    try:
        entry = Entry(metadata=metadata, answer=row["answer"])
        entry.prompt = row["prompt"]
        return float(score_answer(prediction, entry))
    except Exception:
        return None


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    return encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]


def _row_dict(row):
    return row.to_dict() if hasattr(row, "to_dict") else dict(row)


def _metadata(row):
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def _task(row, metadata):
    return (
        metadata.get("source_dataset")
        or metadata.get("task_name")
        or metadata.get("rg_task")
        or row.get("task")
        or metadata.get("_task")
        or metadata.get("task")
    )


def _key(x):
    return str(x).lower().replace("/", "_").replace(" ", "_").replace(".", "_")[:60]


def group_reward_id(rows_by_task, group, spec, max_length):
    """Identity of fixed per-member samples and explicit evaluation weights."""
    group.require_members(rows_by_task)
    payload = [group.identifier, {t: reward_id(rows_by_task[t], spec, max_length)
                                 for t in group.tasks}]
    return "group_reward@v1:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def group_reward(model, tokenizer, rows_by_task, group, spec, max_length):
    """Evaluate n_eval rows PER member, retaining member scores and counts.

    group.weights are evaluation weights. Pass an equally weighted TaskGroup for
    a macro average even when training uses unequal proportions. A member with no
    scorable rows raises instead of silently changing the group being evaluated.
    """
    group.require_members(rows_by_task)
    metrics = {}
    for task in group.tasks:
        result = free_gen_reward(model, tokenizer, rows_by_task[task], spec, max_length)
        if result["reward"] is None:
            raise RuntimeError(f"No scorable intrinsic reward rows for {task}")
        metrics[f"reward/{task}"] = result["reward"]
        metrics[f"reward_examples/{task}"] = result["reward_examples"]
    metrics["reward"] = sum(w * metrics[f"reward/{t}"] for t, w in zip(group.tasks, group.weights))
    return metrics
