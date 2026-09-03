import random
import json
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'Add structural validation of a document against a small schema.',
 'hypothesis': 'S42',
 'changes': 'Ask which path in a nested document violates a stated schema.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1755065315,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class S42SchemaValidationConfig(Config):
    n_keys: int = 4
    depth: int = 2
    valid_frac: float = 0.15

    def apply_difficulty(self, level):
        self.n_keys = sround(self.n_keys + level)
        self.depth = sround(self.depth + level)
        self.valid_frac = 0.15 if level < 5 else 0.18


STRING = "str"
INT = "int"
NUM = "num"
OBJ = "obj"
ARR = "arr"


def _spec_leaf(rnd, key, kind):
    if kind == INT:
        lo = rnd.randint(-5, 10)
        hi = lo + rnd.randint(1, 6)
        return {"key": key, "kind": INT, "lo": lo, "hi": hi}
    if kind == NUM:
        lo = rnd.randint(-5, 10)
        hi = lo + rnd.randint(1, 6)
        return {"key": key, "kind": NUM, "lo": lo, "hi": hi}
    return {"key": key, "kind": STRING}


_WORDS = ["name", "size", "count", "items", "meta", "value", "status", "list",
          "score", "depth", "price", "rank", "node", "edge", "flag",
          "weight", "limit", "total", "field", "index", "width", "height", "max",
          "min", "offset", "length", "type", "mode", "order", "label", "id", "key",
          "pair", "row", "col", "unit", "rate", "sum", "mean", "temp", "speed"]


def _rand_key(rnd, used):
    for _ in range(20):
        w = rnd.choice(_WORDS)
        if rnd.random() < 0.3:
            w = w + str(rnd.randint(0, 9))
        if w not in used:
            used.add(w)
            return w
    w = "f" + str(len(used))
    used.add(w)
    return w


def _build_spec(rnd, depth, n_keys, used=None):
    # returns a list of entries; objects nest by key, arrays hold exactly one
    # scalar element so the doc path stays well-defined (no nested arrays/objects)
    if used is None:
        used = set()
    entries = []
    for i in range(n_keys):
        key = _rand_key(rnd, used)
        if depth > 1 and rnd.random() < max(0.05, 0.35 * (depth - 1) / depth):
            nested = _build_spec(rnd, depth - 1, max(1, n_keys - 1), used)
            entries.append({"key": key, "kind": OBJ, "children": nested})
        elif depth > 1 and rnd.random() < max(0.05, 0.25 * (depth - 1) / depth):
            kind = rnd.choice([INT, NUM, STRING])
            child = _spec_leaf(rnd, _rand_key(rnd, used), kind)
            entries.append({"key": key, "kind": ARR, "children": [child]})
        else:
            kind = rnd.choice([INT, NUM, STRING])
            entries.append(_spec_leaf(rnd, key, kind))
    return entries


def _valid_value(rnd, spec_entry):
    kind = spec_entry["kind"]
    if kind == OBJ:
        return {c["key"]: _valid_value(rnd, c) for c in spec_entry["children"]}
    if kind == ARR:
        return [_valid_value(rnd, spec_entry["children"][0]) for _ in range(rnd.randint(1, 3))]
    if kind == INT:
        return rnd.randint(spec_entry["lo"], spec_entry["hi"])
    if kind == NUM:
        return float(rnd.randint(spec_entry["lo"], spec_entry["hi"]))
    return f"s{rnd.randint(0, 9)}"


def _valid_doc(rnd, spec):
    return {e["key"]: _valid_value(rnd, e) for e in spec}


def _pick_target(rnd, spec, doc, depth_avail):
    """Choose a path (list of keys/indices) to a leaf, descending through the
    actual document. Returns (path, spec_entry)."""
    path = []
    cur_spec = spec
    cur_doc = doc
    while depth_avail > 0 and rnd.random() < 0.5:
        holders = [e for e in cur_spec if e.get("kind") in (OBJ, ARR)]
        if not holders:
            break
        chosen = rnd.choice(holders)
        key = chosen["key"]
        if chosen["kind"] == OBJ:
            path.append(key)
            cur_doc = cur_doc[key]
            cur_spec = chosen["children"]
        else:
            # arrays hold a single scalar spec; target an element by index
            path.append(key)
            arr = cur_doc[key]
            if not arr:
                break
            idx = rnd.randrange(len(arr))
            path.append(idx)
            return path, chosen["children"][0]
        depth_avail -= 1
    target = rnd.choice(cur_spec)
    final_key = target["key"]
    path.append(final_key)
    return path, target


def _mutate_scalar_at(rnd, container, key, target):
    kind = target["kind"]
    if kind == STRING:
        container[key] = rnd.randint(-100, 100)
        return f"key '{key}' must be a string, got {container[key]}"
    vtype = rnd.choice(["range", "type"])
    if vtype == "range":
        if rnd.random() < 0.5:
            container[key] = target["hi"] + rnd.randint(1, 4)
        else:
            container[key] = target["lo"] - rnd.randint(1, 4)
        return (f"key '{key}' must be a number in [{target['lo']}, {target['hi']}], "
                f"got {container[key]}")
    container[key] = f"z{rnd.randint(0, 9)}"
    return f"key '{key}' must be a number, got '{container[key]}'"


def _mutate_leaf(rnd, doc, path, target):
    """Mutate the final leaf through the path, returning description."""
    if isinstance(path[-1], int):
        # array element target: key of parent path is path[-2]
        parent = doc
        for p in path[:-2]:
            parent = parent[p]
        parent_list = parent[path[-2]]
        idx = path[-1]
        kind = target["kind"]
        if kind in (OBJ, ARR):
            parent_list[idx] = rnd.randint(-100, 100)
            return f"array element {idx} must be an object/array, got integer"
        if kind == STRING:
            parent_list[idx] = rnd.randint(-100, 100)
            return f"array element {idx} must be a string, got {parent_list[idx]}"
        vtype = rnd.choice(["range", "type"])
        if vtype == "range":
            parent_list[idx] = target["hi"] + rnd.randint(1, 4)
            return (f"array element {idx} must be a number in "
                    f"[{target['lo']}, {target['hi']}], got {parent_list[idx]}")
        parent_list[idx] = f"z{rnd.randint(0, 9)}"
        return f"array element {idx} must be a number, got '{parent_list[idx]}'"

    cur = doc
    for p in path[:-1]:
        cur = cur[p]
    key = path[-1]
    kind = target["kind"]
    if kind == OBJ or kind == ARR:
        cur[key] = rnd.randint(-100, 100)
        return f"key '{key}' must be a {'object' if kind == OBJ else 'array'}, got integer"
    return _mutate_scalar_at(rnd, cur, key, target)


def _insert_violation(rnd, doc, spec, depth_avail):
    """Return (path_list, description). The doc starts fully valid; we corrupt
    exactly one leaf so there is a single, uniquely-identifiable violation."""
    for attempt in range(40):
        path, target = _pick_target(rnd, spec, doc, depth_avail)
        try:
            desc = _mutate_leaf(rnd, doc, path, target)
            return path, desc
        except (KeyError, IndexError, TypeError):
            continue
    raise RuntimeError("could not place violation")


def _render_schema(spec, indent=0):
    lines = []
    pad = "  " * indent
    for e in spec:
        key = e["key"]
        if e["kind"] == OBJ:
            lines.append(f"{pad}- {key}: object {{")
            lines.append(_render_schema(e["children"], indent + 2))
            lines.append(f"{pad}  }}")
        elif e["kind"] == ARR:
            lines.append(f"{pad}- {key}: array of:")
            lines.append(_render_schema(e["children"], indent + 2))
        elif e["kind"] == STRING:
            lines.append(f"{pad}- {key}: string")
        elif e["kind"] == INT:
            lines.append(f"{pad}- {key}: integer in [{e['lo']}, {e['hi']}]")
        else:
            lines.append(f"{pad}- {key}: number in [{e['lo']}, {e['hi']}]")
    return "\n".join(lines)


def _path_str(path):
    return ".".join(str(p) for p in path)


class SchemaValidation(Task):
    config_cls = S42SchemaValidationConfig

    def generate_entry(self):
        spec = _build_spec(random, self.config.depth, self.config.n_keys)
        doc = _valid_doc(random, spec)

        desc = None
        if random.random() >= self.config.valid_frac:
            path, desc = _insert_violation(random, doc, spec, self.config.depth)
            answer = _path_str(path)
        else:
            answer = "valid"
        metadata = edict({
            "schema": _render_schema(spec),
            "doc": json.dumps(doc),
            "violation_desc": desc,
            "answer": answer,
        })
        metadata.payload = {"schema": metadata.schema, "doc": json.dumps(doc, indent=1)}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        schema = metadata.payload["schema"]
        doc = metadata.payload["doc"]
        return (
            f"Schema (keys are validated in this order):\n{schema}\n\n"
            f"JSON document:\n{doc}\n\n"
            "Validate the document against the schema. Keys are checked in the order the schema "
            "lists them, and array elements by index. Report the dotted path of the FIRST "
            "violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        if a.lower() == "valid":
            return 1.0 if entry.answer == "valid" else 0.0
        # compare canonical dotted path
        return 1.0 if a == entry.answer else 0.0
