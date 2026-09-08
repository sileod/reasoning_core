import json
import numbers
import time
from pathlib import Path

from transformers import TrainerCallback


SCHEMA_VERSION = 1
PERF_SUFFIXES = ("_runtime", "_samples_per_second", "_steps_per_second")


class LocalMetricsSink:
    def __init__(self, path, run_hash, group_id, script_args=None, eff_batch=None):
        self.path = Path(path)
        self.meta_path = self.path.with_name("metrics.meta.json")
        self.run_hash = run_hash
        self.group_id = group_id
        self.script_args = script_args
        self.eff_batch = eff_batch
        self.max_steps = None
        self._meta_written = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_meta(self, max_steps=None):
        self.max_steps = _safe_int(max_steps)
        meta = {
            "run_hash": self.run_hash,
            "group_id": self.group_id,
            "schema_version": SCHEMA_VERSION,
            "args": _args_payload(self.script_args),
            "eff_batch": self.eff_batch,
            "max_steps": self.max_steps,
            "created": time.time(),
        }
        self.meta_path.write_text(json.dumps(meta, sort_keys=True) + "\n")
        self._meta_written = True

    def record(self, metrics, kind, global_step=None, max_steps=None, stage=None, task=None, level=None):
        cleaned = {
            str(k): _json_value(v)
            for k, v in (metrics or {}).items()
            if _json_value(v) is not _SKIP and not _is_perf_key(str(k))
        }
        if not cleaned:
            return

        step = _safe_int(global_step)
        max_steps = _safe_int(max_steps if max_steps is not None else self.max_steps)
        if stage is None and self.script_args is not None:
            stage = getattr(self.script_args, "stage_name", None)
        row = {
            "run_hash": self.run_hash,
            "stage": stage or "unknown",
            "kind": kind,
            "global_step": step,
            "max_steps": max_steps,
            "progress": (step / max_steps) if step is not None and max_steps else None,
            "wall_time": time.time(),
            "metrics": cleaned,
        }
        if task is not None:
            row["task"] = str(task)
        if level is not None:
            row["level"] = str(level)

        with self.path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()


class LocalMetricsCallback(TrainerCallback):
    def __init__(self, sink):
        self.sink = sink

    def on_train_begin(self, args, state, control, **kwargs):
        if self.sink._meta_written:
            return
        self.sink.write_meta(max_steps=getattr(state, "max_steps", None))

    def on_log(self, args, state, control, logs=None, **kwargs):
        for kind, metrics, task, level in split_log_records(logs or {}):
            self.sink.record(
                metrics,
                kind=kind,
                global_step=getattr(state, "global_step", None),
                max_steps=getattr(state, "max_steps", None),
                task=task,
                level=level,
            )


def split_log_records(logs):
    rows = []
    train_metrics = {}
    eval_main_metrics = {}
    aux_metrics = {}

    for raw_key, value in logs.items():
        key = str(raw_key)
        value = _json_value(value)
        if _is_perf_key(key) or value is _SKIP:
            continue

        if key.startswith("downstream_eval/") or key.startswith("downstream_eval_"):
            continue
        if key.startswith("eval_aux_tl/") or key.startswith("eval_aux_tl_"):
            parsed = _parse_aux_key(key)
            if parsed is not None:
                task, level, metric = parsed
                aux_metrics.setdefault((task, level), {})[metric] = value
            continue
        if key.startswith("final_aux_tl/") or key.startswith("final_aux_tl_"):
            parsed = _parse_aux_key(key, prefixes=("final_aux_tl/", "final_aux_tl_"))
            if parsed is not None:
                task, level, metric = parsed
                aux_metrics.setdefault((task, level), {})[metric] = value
            continue
        if key.startswith("eval_main_") or key.startswith("eval/main_"):
            metric = _strip_first_prefix(key, ("eval_main_", "eval/main_"))
            if metric and not _is_perf_key(metric):
                eval_main_metrics[_clean_metric(metric)] = value
            continue
        if key.startswith("eval_") or key.startswith("eval/"):
            metric = _strip_first_prefix(key, ("eval_", "eval/"))
            if metric and not _is_perf_key(metric):
                eval_main_metrics[_clean_metric(metric)] = value
            continue
        if key.startswith("final_main_") or key.startswith("final/main_"):
            metric = _strip_first_prefix(key, ("final_main_", "final/main_"))
            if metric and not _is_perf_key(metric):
                eval_main_metrics[_clean_metric(metric)] = value
            continue
        if key.startswith("final_") or key.startswith("final/"):
            metric = _strip_first_prefix(key, ("final_", "final/"))
            if metric and not _is_perf_key(metric):
                eval_main_metrics[_clean_metric(metric)] = value
            continue
        if key in {"loss", "grad_norm", "learning_rate", "epoch"} or key.startswith("train_") or key.startswith("train/"):
            train_metrics[_clean_metric(_strip_first_prefix(key, ("train_", "train/")) or key)] = value

    if train_metrics:
        rows.append(("train", train_metrics, None, None))
    if eval_main_metrics:
        rows.append(("eval_main", eval_main_metrics, None, None))
    for (task, level), metrics in sorted(aux_metrics.items()):
        rows.append(("eval_aux_tl", metrics, task, level))
    return rows


def load(run_hash, root="checkpoints"):
    import pandas as pd

    base = Path(root) / run_hash
    meta = json.loads((base / "metrics.meta.json").read_text())
    rows = []
    with (base / "metrics.jsonl").open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return meta, pd.DataFrame(rows)


def _args_payload(args):
    payload = {}
    if args is not None:
        for key, value in sorted(vars(args).items()):
            clean_value = _json_value(value)
            if clean_value is not _SKIP:
                payload[key] = clean_value
    payload["completion_only_loss"] = True
    return payload


def _parse_aux_key(key, prefixes=("eval_aux_tl/", "eval_aux_tl_")):
    rest = _strip_first_prefix(key, prefixes)
    if not rest:
        return None
    marker = "_mean_token_accuracy"
    if rest.endswith(marker):
        group = rest[: -len(marker)]
        metric = "mean_token_accuracy"
    elif rest.endswith("_loss"):
        group = rest[:-5]
        metric = "loss"
    else:
        return None
    if "." in group:
        task, level = group.rsplit(".", 1)
    else:
        task, level = group, ""
    return task, level, metric


def _strip_first_prefix(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def _clean_metric(metric):
    return str(metric).replace("/", "_")


def _is_perf_key(key):
    return str(key).endswith(PERF_SUFFIXES)


_SKIP = object()


def _json_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            return _SKIP
    return _SKIP


def _safe_int(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
