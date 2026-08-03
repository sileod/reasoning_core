"""Versioned local and Hugging Face stream recipes."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import interleave_datasets, load_dataset


FORMATTERS = ("sft_qa_v1", "influence_legacy_v1", "text_v1")


@dataclass(frozen=True)
class StreamSpec:
    source: str
    formatter: str
    split: str = "train"
    config: str | None = None
    cycle: bool = False
    prompt_prefix: str = ""
    task: str | None = None

    def __post_init__(self):
        if self.formatter not in FORMATTERS:
            raise ValueError(f"Unknown formatter {self.formatter!r}; choose from {FORMATTERS}")


def format_row(row, eos_token, formatter, prompt_prefix=""):
    if formatter == "text_v1":
        return {"prompt": "", "completion": row["text"] + eos_token}
    prompt, answer = row["prompt"], row["answer"]
    if formatter == "sft_qa_v1":
        return {
            "prompt": f"{prompt_prefix}Q: {prompt}\nA:",
            "completion": f" {answer}{eos_token}",
        }
    return {"prompt": f"{prompt_prefix}{prompt}\n", "completion": f"{answer}{eos_token}"}


def load_stream(spec, tokenizer, max_length=None, chars_per_token=4.0, max_tokens=None):
    """Load a replayable local or HF stream and apply a versioned formatter."""

    max_chars = max_length * chars_per_token if max_length and max_tokens is None else None
    raw = _raw_stream(spec)
    if spec.task:
        raw = raw.filter(lambda row: row.get("task") == spec.task)

    def format_example(row, index):
        return {
            **format_row(row, tokenizer.eos_token, spec.formatter, spec.prompt_prefix),
            "_source": spec.source,
            "_source_index": index,
        }

    stream = raw.map(format_example, with_indices=True, remove_columns=raw.column_names)
    if max_chars:
        stream = stream.filter(
            lambda row: len(row["prompt"]) + len(row["completion"]) <= max_chars
        )
    if max_tokens:
        stream = stream.filter(lambda row: formatted_length(row, tokenizer) <= max_tokens)
    return stream.repeat(None) if spec.cycle else stream


def mix_streams(main, aux=None, aux_fraction=0.0, seed=42, *, shuffle_buffer):
    """Interleave streams, with ordering made explicit at every call site.

    ``shuffle_buffer=0`` reproduces the legacy influence protocol. A positive
    value enables deterministic buffered shuffling as a deliberate protocol
    choice rather than a hidden default.
    """

    parts = [stream for stream in (main, aux) if stream is not None]
    if len(parts) == 2:
        if not 0 < aux_fraction < 1:
            raise ValueError(f"aux_fraction must be between 0 and 1, got {aux_fraction}")
        stream = interleave_datasets(
            parts, probabilities=[1 - aux_fraction, aux_fraction],
            seed=seed, stopping_strategy="first_exhausted",
        )
    else:
        stream = parts[0]
    return stream.shuffle(seed=seed, buffer_size=shuffle_buffer) if shuffle_buffer else stream


def fraction_for_token_share(target, main_mean_tokens, aux_mean_tokens):
    """Example probability that yields ``target`` auxiliary tokens under packing."""

    if not 0 < target < 1:
        raise ValueError(f"target must be between 0 and 1, got {target}")
    if main_mean_tokens <= 0 or aux_mean_tokens <= 0:
        raise ValueError("mean token lengths must be positive")
    return target * main_mean_tokens / (
        aux_mean_tokens * (1 - target) + target * main_mean_tokens
    )


def steps_for_token_budget(token_budget, aux_ratio, max_length, effective_batch_size):
    total = token_budget * (1 + aux_ratio)
    return max(1, int(total // (max_length * effective_batch_size)))


def ratio_to_fraction(aux_ratio):
    return aux_ratio / (1 + aux_ratio)


def formatted_length(row, tokenizer):
    prompt = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
    completion = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    return len(prompt) + len(completion)


def replay_after(stream_factory, consumed):
    return stream_factory().skip(consumed)


def settle_remote_streams(seconds=1):
    gc.collect()
    time.sleep(seconds)


def _raw_stream(spec):
    path = Path(spec.source).expanduser()
    if path.exists():
        if path.is_dir():
            files = sorted(str(file) for file in path.glob("**/*.parquet"))
            if not files:
                raise ValueError(f"Local stream directory has no Parquet files: {path}")
            return load_dataset("parquet", data_files=files, split=spec.split, streaming=True)
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl", ".parquet"}:
            raise ValueError(f"Local stream must be JSONL/JSON/Parquet, got {path}")
        loader = "parquet" if suffix == ".parquet" else "json"
        return load_dataset(loader, data_files=str(path), split=spec.split, streaming=True)
    return load_dataset(spec.source, spec.config, split=spec.split, streaming=True)
