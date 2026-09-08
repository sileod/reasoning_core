"""Cheap gradient-alignment proxy for paired training influence experiments."""

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from reasoning_core import __version__
from reasoning_core.evaluation.metrics import (
    contrastive_mc_loss,
    contrastive_mc_objective_id,
    evaluating,
)


@dataclass(frozen=True)
class GradientCacheSpec:
    initialization_id: str
    objective_id: str
    max_length: int
    dtype: str = "bfloat16"


@dataclass
class GradientCache:
    spec: GradientCacheSpec
    gradients: dict
    norm: float


@dataclass
class GradientInfluenceResult:
    cosine: float
    dot: float
    task_norm: float
    eval_norm: float
    examples: int
    tokens: int


@dataclass(frozen=True)
class CompletionLoss:
    loss: torch.Tensor
    examples: int
    tokens: int
    skipped_examples: int = 0


@dataclass(frozen=True)
class TaskGradientEstimate:
    mean_cosine: float
    std_cosine: float
    stderr_cosine: float
    aggregate_cosine: float
    batches: tuple[GradientInfluenceResult, ...]


def completion_loss(model, tokenizer, rows, max_length):
    """Token-weighted completion-only loss for formatted prompt/completion rows.

    Tokenization and masking mirror TRL's non-conversational prompt-completion
    preprocessing. Packing is intentionally unsupported.
    """

    encoded, skipped = [], 0
    for row in rows:
        prompt, completion = str(row["prompt"]), str(row["completion"])
        prompt_ids = _ids(tokenizer, prompt)
        input_ids = _ids(tokenizer, prompt + completion)
        if len(input_ids) > max_length or len(input_ids) <= len(prompt_ids):
            skipped += 1
            continue
        encoded.append((input_ids, len(prompt_ids)))
    if not encoded:
        raise RuntimeError(
            f"Completion loss scored no rows ({skipped} overlength or empty completions)"
        )

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id")
    width = max(len(ids) for ids, _ in encoded)
    inputs = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, -100)
    attention = torch.zeros_like(inputs)
    for index, (ids, prompt_length) in enumerate(encoded):
        length = len(ids)
        inputs[index, :length] = torch.tensor(ids, device=device)
        labels[index, prompt_length:length] = inputs[index, prompt_length:length]
        attention[index, :length] = 1
    tokens = int((labels[:, 1:] != -100).sum().item())
    if not tokens:
        raise RuntimeError("Completion loss scored no completion tokens")
    loss = model(input_ids=inputs, attention_mask=attention, labels=labels).loss
    return CompletionLoss(loss, len(encoded), tokens, skipped)


def gradient_objective_id(legs, max_length, temperature=1.0, weights=None):
    """Content ID for an ordered aggregate of contrastive MC benchmark legs."""

    legs = _materialize_legs(legs)
    weights = _weights(legs, weights)
    ids = [contrastive_mc_objective_id(name, examples, max_length, temperature)
           for name, examples in legs]
    payload = {"version": 1, "legs": ids, "weights": weights}
    digest = hashlib.sha256(_json(payload).encode()).hexdigest()[:16]
    return f"aggregate_contrastive_mc@v1:{digest}"


def build_eval_gradient_cache(model, tokenizer, legs, spec, *, temperature=1.0,
                              weights=None, cache_dir=None, model_id=None,
                              model_revision=None, eval_batch_size=1):
    """Build, normalize, and persist one aggregate benchmark gradient."""

    if not spec.initialization_id:
        raise ValueError("initialization_id is required")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    legs = _materialize_legs(legs)
    weights = _weights(legs, weights)
    actual_id = gradient_objective_id(legs, spec.max_length, temperature, weights)
    if spec.objective_id != actual_id:
        raise ValueError(f"objective_id mismatch: expected {actual_id}")
    dtype = _dtype(spec.dtype)
    aggregate, total_examples, total_tokens, leg_manifest = {}, 0, 0, []
    with evaluating(model):
        for (name, examples), weight in zip(legs, weights):
            model.zero_grad(set_to_none=True)
            leg_examples = leg_tokens = leg_skipped = 0
            for start in range(0, len(examples), eval_batch_size):
                chunk = examples[start:start + eval_batch_size]
                try:
                    result = contrastive_mc_loss(
                        model, tokenizer, chunk, spec.max_length, temperature,
                    )
                except RuntimeError as error:
                    if "scored no examples" not in str(error):
                        raise
                    leg_skipped += len(chunk)
                    continue
                (result.loss * result.examples).backward()
                leg_examples += result.examples
                leg_tokens += result.tokens
                leg_skipped += result.skipped_examples
            norm = _model_grad_norm(model)
            if not norm:
                raise RuntimeError(f"Benchmark leg {name!r} produced a zero gradient")
            for parameter_name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    value = parameter.grad.detach().float().cpu().mul_(weight / norm)
                    aggregate.setdefault(parameter_name, torch.zeros_like(value)).add_(value)
            total_examples += leg_examples
            total_tokens += leg_tokens
            leg_manifest.append({
                "name": name, "weight": weight, "gradient_norm": norm,
                "examples": leg_examples, "tokens": leg_tokens,
                "skipped_examples": leg_skipped,
            })
    model.zero_grad(set_to_none=True)
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter_name not in aggregate:
            aggregate[parameter_name] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
    aggregate_norm = math.sqrt(sum(value.square().sum().item()
                                   for value in aggregate.values()))
    if not aggregate_norm:
        raise RuntimeError("Aggregate benchmark gradient has zero norm")
    gradients = {name: (value / aggregate_norm).to(dtype).contiguous()
                 for name, value in aggregate.items()}
    stored_norm = math.sqrt(sum(value.float().square().sum().item()
                                for value in gradients.values()))
    path = _cache_path(spec, cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    save_file(gradients, str(path / "gradients.safetensors"))
    inferred_model_id, inferred_revision = _model_identity(model)
    manifest = {
        "cache_version": 1,
        "spec": asdict(spec),
        "cache_id": path.name,
        "model": model_id or inferred_model_id,
        "model_revision": model_revision or inferred_revision,
        "gradient_norm": stored_norm,
        "examples": total_examples,
        "tokens": total_tokens,
        "parameters": {name: list(value.shape) for name, value in gradients.items()},
        "legs": leg_manifest,
        "library_version": __version__,
        "dependencies": _versions(),
    }
    _atomic_json(path / "manifest.json", manifest)
    device = next(model.parameters()).device
    return GradientCache(
        spec, {name: value.to(device) for name, value in gradients.items()}, stored_norm,
    )


def load_gradient_cache(spec, *, device=None, cache_dir=None):
    """Load and validate a content-addressed gradient cache."""

    path = _cache_path(spec, cache_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("spec") != asdict(spec):
        raise RuntimeError(f"Gradient cache spec mismatch in {path}")
    gradients = load_file(str(path / "gradients.safetensors"), device=str(device or "cpu"))
    shapes = {name: list(value.shape) for name, value in gradients.items()}
    if shapes != manifest.get("parameters"):
        raise RuntimeError(f"Gradient cache parameter manifest mismatch in {path}")
    return GradientCache(spec, gradients, float(manifest["gradient_norm"]))


def score_task_gradient(model, tokenizer, rows, cache, max_length=512):
    """Score one formatted task batch without copying or flattening its gradient."""

    _validate_cache(model, cache, max_length)
    with evaluating(model):
        model.zero_grad(set_to_none=True)
        result = completion_loss(model, tokenizer, rows, max_length)
        result.loss.backward()
    dot, task_sq = _alignment(model, cache)
    task_norm = math.sqrt(task_sq)
    cosine = dot / (cache.norm * task_norm) if cache.norm and task_norm else 0.0
    return GradientInfluenceResult(
        cosine, dot, task_norm, cache.norm, result.examples, result.tokens,
    )


def score_task(model, tokenizer, batch_factory, cache, *, batches=8,
               max_length=512, seed=0):
    """Score deterministic batches independently and as one accumulated gradient.

    ``batch_factory(seed)`` should return already formatted rows. An iterable of
    batches is also accepted; in that case its existing order is used.
    """

    if batches < 1:
        raise ValueError("batches must be positive")
    _validate_cache(model, cache, max_length)
    if callable(batch_factory):
        materialized = []
        for index in range(batches):
            batch_seed = seed + index
            with _deterministic_seed(batch_seed):
                materialized.append(list(batch_factory(batch_seed)))
    else:
        materialized = [list(batch) for batch in batch_factory][:batches]
        if len(materialized) != batches:
            raise ValueError(f"Expected {batches} batches, got {len(materialized)}")

    results = []
    model.zero_grad(set_to_none=True)
    with evaluating(model):
        for rows in materialized:
            loss = completion_loss(model, tokenizer, rows, max_length)
            dot, square = _backward_stats(loss.loss, model, cache)
            norm = math.sqrt(square)
            cosine = dot / (cache.norm * norm) if cache.norm and norm else 0.0
            results.append(GradientInfluenceResult(
                cosine, dot, norm, cache.norm, loss.examples, loss.tokens,
            ))
    aggregate_dot, aggregate_sq = _alignment(model, cache)
    aggregate_norm = math.sqrt(aggregate_sq)
    aggregate_cosine = (
        aggregate_dot / (cache.norm * aggregate_norm)
        if cache.norm and aggregate_norm else 0.0
    )
    values = [result.cosine for result in results]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) \
        if len(values) > 1 else 0.0
    return TaskGradientEstimate(
        mean, std, std / math.sqrt(len(values)), aggregate_cosine, tuple(results),
    )


def _backward_stats(loss, model, cache):
    """Observe this backward's incoming gradients while ``.grad`` accumulates."""

    dots, squares, handles = [], [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name in cache.gradients:
            cached = cache.gradients[name]

            def observe(gradient, cached=cached):
                value = gradient.float()
                dots.append((value * cached.float()).sum())
                squares.append(value.square().sum())

            handles.append(parameter.register_hook(observe))
    try:
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    return _scalar_sum(dots), _scalar_sum(squares)


def _alignment(model, cache):
    dots, squares = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name not in cache.gradients:
            raise RuntimeError(f"Cached gradient is missing parameter {name!r}")
        value = parameter.grad.float()
        dots.append((value * cache.gradients[name].float()).sum())
        squares.append(value.square().sum())
    return _scalar_sum(dots), _scalar_sum(squares)


def _model_grad_norm(model):
    return math.sqrt(sum(parameter.grad.detach().float().square().sum().item()
                         for parameter in model.parameters() if parameter.grad is not None))


def _validate_cache(model, cache, max_length):
    if max_length != cache.spec.max_length:
        raise ValueError(
            f"Task max_length={max_length} differs from cached max_length={cache.spec.max_length}"
        )
    parameters = dict(model.named_parameters())
    required = {name for name, parameter in parameters.items() if parameter.requires_grad}
    missing = required - cache.gradients.keys()
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise ValueError(f"Cached gradient is missing trainable parameters: {preview}")
    for name, gradient in cache.gradients.items():
        if name not in parameters or parameters[name].shape != gradient.shape:
            raise ValueError(f"Cached parameter {name!r} does not match the model")
        if gradient.device != parameters[name].device:
            raise ValueError(
                f"Cached gradient {name!r} is on {gradient.device}, model is on "
                f"{parameters[name].device}; reload the cache on the model device"
            )


def _materialize_legs(legs):
    items = legs.items() if hasattr(legs, "items") else legs
    result = [(str(name), tuple(examples)) for name, examples in items]
    if not result or len({name for name, _ in result}) != len(result):
        raise ValueError("Benchmark legs must be non-empty and uniquely named")
    return result


def _weights(legs, weights):
    if weights is None:
        return [1.0] * len(legs)
    if hasattr(weights, "items"):
        weights = [weights[name] for name, _ in legs]
    values = [float(value) for value in weights]
    if len(values) != len(legs) or any(value < 0 for value in values) or not any(values):
        raise ValueError("weights must be non-negative, non-zero, and match the legs")
    return values


def _cache_path(spec, cache_dir):
    root = Path(cache_dir or Path.home() / ".cache/reasoning_core/gradient_influence")
    cache_id = hashlib.sha256(_json(asdict(spec)).encode()).hexdigest()[:20]
    return root.expanduser() / cache_id


def _dtype(name):
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"Unknown torch dtype {name!r}") from error
    if not dtype.is_floating_point:
        raise ValueError("Gradient cache dtype must be floating point")
    return dtype


def _ids(tokenizer, text):
    encoded = tokenizer(text)
    return list(encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"])


def _scalar_sum(values):
    return float(torch.stack(values).sum().item()) if values else 0.0


def _model_identity(model):
    config = getattr(model, "config", None)
    return (getattr(config, "_name_or_path", None), getattr(config, "_commit_hash", None))


def _versions():
    result = {}
    for package in ("torch", "transformers", "safetensors"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def _deterministic_seed(seed):
    python_state, numpy_state, torch_state = (
        random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
    )
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
