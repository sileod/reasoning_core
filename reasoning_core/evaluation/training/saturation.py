"""Versioned, resumable answer-token saturation curves."""

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import TrainerCallback


SATURATION_VERSION = 1


@dataclass(frozen=True)
class SaturationCurveSpec:
    """Protocol for periodic teacher-forced answer-token accuracy."""

    every_steps: int = 50
    n_eval: int = 40
    batch_size: int = 8

    def __post_init__(self):
        if min(self.every_steps, self.n_eval, self.batch_size) < 1:
            raise ValueError("every_steps, n_eval, and batch_size must be positive")


def saturation_id(rows, spec, max_length):
    payload = {
        "version": SATURATION_VERSION,
        "spec": asdict(spec),
        "max_length": max_length,
        "rows": [_row_dict(row) for row in rows[:spec.n_eval]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"saturation@v{SATURATION_VERSION}:{hashlib.sha256(encoded.encode()).hexdigest()[:12]}"


@torch.no_grad()
def answer_token_accuracy(model, tokenizer, rows, max_length, batch_size=8):
    """Teacher-forced argmax accuracy over answer tokens, evaluated in batches."""

    encoded = []
    eos = tokenizer.eos_token or ""
    for raw in rows:
        row = _row_dict(raw)
        prompt = _token_ids(tokenizer, f"{row['prompt']}\n")
        answer = _token_ids(tokenizer, f"{row['answer']}{eos}")
        if prompt and answer and len(prompt) + len(answer) <= max_length:
            encoded.append((prompt + answer, len(prompt), len(answer)))
    if not encoded:
        return 0.0
    encoded.sort(key=lambda item: len(item[0]))

    was_training = model.training
    model.eval()
    correct = total = 0
    try:
        device = next(model.parameters()).device
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id or 0
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            width = max(len(ids) for ids, _, _ in batch)
            input_ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for index, (ids, _, _) in enumerate(batch):
                input_ids[index, :len(ids)] = torch.tensor(ids, device=device)
                attention_mask[index, :len(ids)] = 1
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for index, (_, prompt_length, answer_length) in enumerate(batch):
                first = prompt_length - 1
                prediction = logits[index, first:first + answer_length].argmax(-1)
                gold = input_ids[index, prompt_length:prompt_length + answer_length]
                correct += (prediction == gold).sum().item()
                total += answer_length
    finally:
        model.train(was_training)
    return correct / total


def derive_saturation(curve):
    if not curve:
        return {}
    accuracies = [accuracy for _, accuracy in curve]
    final = accuracies[-1]
    threshold = 0.95 * final
    saturation_step = next(
        (step for step, accuracy in curve if accuracy >= threshold), curve[-1][0],
    )
    return {
        "curve": curve,
        "acc0": curve[0][1],
        "acc_final": final,
        "auc": sum(accuracies) / len(accuracies),
        "sat_step": saturation_step,
        "n_points": len(curve),
    }


class SaturationCurveCallback(TrainerCallback):
    """Evaluate and durably record a curve at step zero and fixed intervals."""

    def __init__(self, tokenizer, rows, spec, max_length, path):
        self.tokenizer = tokenizer
        self.rows = tuple(rows[:spec.n_eval])
        self.spec = spec
        self.max_length = max_length
        self.path = Path(path)
        self.curve = _load_curve(self.path)

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.global_step == 0:
            self._record(0, model, optimizer)

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.global_step % self.spec.every_steps == 0:
            self._record(state.global_step, model, optimizer)

    def result_metrics(self):
        return {"saturation": derive_saturation(self.curve)}

    def _record(self, step, model, optimizer):
        if model is None or any(existing == step for existing, _ in self.curve):
            return
        context = nullcontext()
        if optimizer is not None and hasattr(optimizer, "eval"):
            from reasoning_core.evaluation.training.arm import optimizer_eval_mode
            context = optimizer_eval_mode(optimizer)
        with context:
            accuracy = answer_token_accuracy(
                model, self.tokenizer, self.rows, self.max_length, self.spec.batch_size,
            )
        self.curve.append([int(step), float(accuracy)])
        self.curve.sort(key=lambda point: point[0])
        _write_curve(self.path, self.curve)


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"])


def _row_dict(row):
    return row.to_dict() if hasattr(row, "to_dict") else dict(row)


def _load_curve(path):
    try:
        value = json.loads(path.read_text())
        return [[int(step), float(accuracy)] for step, accuracy in value.get("curve", ())]
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return []


def _write_curve(path, curve):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"curve": curve}, sort_keys=True) + "\n")
    os.replace(temporary, path)
