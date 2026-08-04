"""Ordered, content-addressed benchmark batteries."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from reasoning_core.training.evals import (
    evaluate_generation,
    evaluate_lm_nll,
    evaluate_mcq,
    evaluate_qa_nll,
    load_eval_suite,
)


KINDS = ("qa_nll", "lm_nll", "mcq", "generation")


@dataclass(frozen=True)
class EvalLeg:
    name: str
    path: str
    kind: str
    output_key: str
    limit: int | None = None
    max_new_tokens: int = 64
    accuracy_key: str | None = None
    margin_key: str | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown evaluation kind {self.kind!r}; choose from {KINDS}")

    @property
    def identifier(self):
        config = json.dumps({
            "kind": self.kind, "output_key": self.output_key, "limit": self.limit,
            "max_new_tokens": self.max_new_tokens,
            "accuracy_key": self.accuracy_key, "margin_key": self.margin_key,
        }, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(Path(self.path).expanduser().read_bytes() + config).hexdigest()[:12]
        return f"{self.name}/{self.kind}@v1:{digest}"


@dataclass(frozen=True)
class EvalBattery:
    name: str
    legs: tuple[EvalLeg, ...]
    max_length: int = 512

    def __post_init__(self):
        names = [leg.name for leg in self.legs]
        if len(names) != len(set(names)):
            raise ValueError("Evaluation leg names must be unique")

    @property
    def identifier(self):
        payload = {"name": self.name, "max_length": self.max_length,
                   "legs": [leg.identifier for leg in self.legs]}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.name}/battery@v1:{digest}"


@dataclass(frozen=True)
class BatteryResult:
    metrics: dict[str, float]
    legs: dict[str, dict]


def evaluate_battery(model, tokenizer, battery, eos_token):
    """Evaluate every leg in manifest order and return flat metrics plus details."""

    metrics, details = {}, {}
    for leg in battery.legs:
        if leg.kind == "lm_nll":
            texts = _load_texts(leg.path, leg.limit)
            result = evaluate_lm_nll(model, tokenizer, texts, battery.max_length)
            metrics[leg.output_key] = result["nll"]
        else:
            suite = load_eval_suite(
                leg.path, eos_token, name=leg.name, limit=leg.limit,
            )
            if leg.kind == "qa_nll":
                result = evaluate_qa_nll(
                    model, tokenizer, suite.examples, battery.max_length,
                )
                metrics[leg.output_key] = result["nll"]
            elif leg.kind == "mcq":
                result = evaluate_mcq(
                    model, tokenizer, suite.examples, battery.max_length,
                )
                _add_mcq_metrics(metrics, leg, result)
            else:
                result = evaluate_generation(
                    model, tokenizer, suite.examples, battery.max_length,
                    leg.max_new_tokens,
                )
                metrics[leg.output_key] = _required(leg, result, "score")
        details[leg.name] = result
    return BatteryResult(metrics, details)


def paper_battery(data_dir="data_cache", max_length=512):
    """The ordered held-out battery used by the influence paper."""

    manifest = Path(__file__).with_name("paper_battery.json")
    return load_battery_manifest(manifest, data_dir, max_length)


def load_battery_manifest(path, data_dir=None, max_length=None):
    """Load an arbitrary battery from a JSON object with an ordered ``legs`` list."""

    path = Path(path).expanduser()
    payload = json.loads(path.read_text())
    root = Path(data_dir) if data_dir is not None else path.parent
    legs = tuple(EvalLeg(
        **{**row, "path": str(root / row["path"])
           if not Path(row["path"]).is_absolute() else row["path"]}
    ) for row in payload["legs"])
    return EvalBattery(
        payload.get("name", path.stem), legs,
        max_length if max_length is not None else payload.get("max_length", 512),
    )


def _add_mcq_metrics(metrics, leg, result):
    stem = leg.output_key.removesuffix("_nll")
    metrics[leg.output_key] = _required(leg, result, "gold_nll")
    accuracy_key = leg.accuracy_key or f"{stem}_mc_cloze_acc"
    metrics[accuracy_key] = _required(leg, result, "accuracy")
    if result["margin"] is not None:
        metrics[leg.margin_key or f"{stem}_mc_cloze_margin"] = result["margin"]


def _required(leg, result, key):
    value = result[key]
    if value is None:
        name = leg.name if isinstance(leg, EvalLeg) else leg
        raise RuntimeError(f"Evaluation leg {name!r} produced no {key}")
    return value


def _load_texts(path, limit):
    texts = []
    with Path(path).expanduser().open() as file:
        for line in file:
            row = json.loads(line)
            if row.get("text") is not None:
                texts.append(str(row["text"]))
            if limit and len(texts) >= limit:
                break
    return texts
