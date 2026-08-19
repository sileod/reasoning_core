"""Ordered, content-addressed benchmark batteries."""

import hashlib
import json
import os
import zipfile
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
    max_chars: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown evaluation kind {self.kind!r}; choose from {KINDS}")

    @property
    def metric_keys(self):
        if self.kind != "mcq":
            return (self.output_key,)
        stem = self.output_key.removesuffix("_nll")
        return (
            self.output_key,
            self.accuracy_key or f"{stem}_mc_cloze_acc",
            self.margin_key or f"{stem}_mc_cloze_margin",
        )

    @property
    def identifier(self):
        config = json.dumps({
            "kind": self.kind, "output_key": self.output_key, "limit": self.limit,
            "max_new_tokens": self.max_new_tokens,
            "accuracy_key": self.accuracy_key, "margin_key": self.margin_key,
            "max_chars": self.max_chars,
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
        keys = [key for leg in self.legs for key in leg.metric_keys]
        if len(keys) != len(set(keys)):
            raise ValueError("Evaluation metric keys must be unique")

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
            texts = _load_texts(leg.path, leg.limit, leg.max_chars)
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


DATA_DIR = os.environ.get("EVAL_DATA_DIR", "data_cache")
# Leg identity is sha256 of the file bytes, so legs ship rather than rebuild. Outside the package:
# neither wheel nor sdist carries them.
_ARCHIVE = Path(__file__).resolve().parents[2] / "eval_data" / "battery_legs.zip"


def ensure_eval_data(data_dir=DATA_DIR):
    """Unpack the shipped legs into ``data_dir``; no-op if present or not shipped."""

    root = Path(data_dir).expanduser()
    if _ARCHIVE.exists():
        with zipfile.ZipFile(_ARCHIVE) as archive:
            missing = [n for n in archive.namelist() if not (root / n).exists()]
            if missing:
                archive.extractall(root, members=missing)
                print(f"[battery] unpacked {len(missing)} eval legs into {root}")
    return root


def paper_battery(data_dir=DATA_DIR, max_length=512):
    """The ordered held-out battery used by the influence paper."""

    manifest = Path(__file__).with_name("paper_battery.json")
    return load_battery_manifest(manifest, data_dir, max_length)


def load_battery_manifest(path, data_dir=None, max_length=None):
    """Load an arbitrary battery from a JSON object with an ordered ``legs`` list."""

    path = Path(path).expanduser()
    payload = json.loads(path.read_text())
    root = ensure_eval_data(DATA_DIR if data_dir is None else data_dir)
    legs = tuple(EvalLeg(
        **{**row, "path": str(root / row["path"])
           if not Path(row["path"]).is_absolute() else row["path"]}
    ) for row in payload["legs"])
    return EvalBattery(
        payload.get("name", path.stem), legs,
        max_length if max_length is not None else payload.get("max_length", 512),
    )


def _add_mcq_metrics(metrics, leg, result):
    nll_key, accuracy_key, margin_key = leg.metric_keys
    metrics[nll_key] = _required(leg, result, "gold_nll")
    metrics[accuracy_key] = _required(leg, result, "accuracy")
    if result["margin"] is not None:
        metrics[margin_key] = result["margin"]


def _required(leg, result, key):
    value = result[key]
    if value is None:
        name = leg.name if isinstance(leg, EvalLeg) else leg
        raise RuntimeError(f"Evaluation leg {name!r} produced no {key}")
    return value


def _load_texts(path, limit, max_chars=None):
    texts = []
    with Path(path).expanduser().open() as file:
        for line in file:
            row = json.loads(line)
            if row.get("text") is not None:
                text = str(row["text"])
                texts.append(text[:max_chars] if max_chars else text)
            if limit and len(texts) >= limit:
                break
    return texts
