import json

from datasets import IterableDataset


def add_control_args(parser):
    parser.add_argument("--aux_task", type=str, default="")
    parser.add_argument("--aux_mode", type=str, default="")
    parser.add_argument("--aux_level", type=str, default="")


def row_filter(row, *, task=None, mode=None, level=None):
    if task and _value(row, "task") != str(task):
        return False
    if mode and _value(row, "mode") != str(mode):
        return False
    if level and _value(row, "level") != str(level):
        return False
    return True


def wrap_aux_dataset_for_control(ds, args):
    spec = control_spec(args)
    if ds is None or not any(spec.values()):
        return ds, control_spec(args)

    def gen():
        for row in ds:
            if row_filter(row, task=spec["aux_task"], mode=spec["aux_mode"], level=spec["aux_level"]):
                yield row

    return IterableDataset.from_generator(gen), spec


def control_spec(args):
    return {
        "aux_task": _empty_to_none(getattr(args, "aux_task", "")),
        "aux_mode": _empty_to_none(getattr(args, "aux_mode", "")),
        "aux_level": _empty_to_none(getattr(args, "aux_level", "")),
    }


def _value(row, key):
    value = row.get(key) if isinstance(row, dict) else None
    if value in (None, ""):
        meta = _metadata(row)
        value = meta.get(key) or meta.get(f"_{key}")
    return None if value in (None, "") else str(value)


def _metadata(row):
    raw = row.get("metadata", {}) if isinstance(row, dict) else {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def _empty_to_none(value):
    value = str(value).strip()
    return value or None
