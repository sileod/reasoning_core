"""Lazy task discovery, loading, and scorer dispatch."""

import ast
import functools
import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from difflib import get_close_matches
from pathlib import Path

from appdirs import user_cache_dir
from easydict import EasyDict as edict
from inflection import underscore


_REGISTRY = {}
_PACKAGE_NAME = __package__
_TASKS_PATH = Path(__file__).parent / "tasks"
_CACHE_PATH = Path(user_cache_dir("reasoning_core")) / "task_registry.json"
_CACHE_VERSION = 1

COLLECTIONS = {
    "procedural_warmup": ("_procedural_warmup", "ProceduralWarmup"),
    "reasoning_gym": ("_reasoning_gym", "Reasoning_Gym"),
    "synlogic": ("_synlogic", "Synlogic"),
}
DEPRECATED = {"symbolic_arithmetics", "graph_node_centrality"}
IGNORED = DEPRECATED | {"reasonining_gym", "count_elements"}


def register_dataset(name, dataset_cls):
    _REGISTRY[name] = dataset_cls


def prepr_task_name(name):
    return underscore(name)


def _parse_task_file(path, relative):
    tree = ast.parse(path.read_text(), filename=str(relative))
    found = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        kind = "dev" if "DevTask" in bases else "task" if "Task" in bases else None
        if kind is None:
            continue
        name = prepr_task_name(node.name)
        for item in node.body:
            if (isinstance(item, ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "task_name"
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)):
                name = item.value.value
                break
        found.append([name, node.name, kind])
    return found


def _load_discovery_cache(path, root):
    try:
        data = json.loads(path.read_text())
        valid = data.get("version") == _CACHE_VERSION and data.get("root") == str(root.resolve())
        return data.get("files", {}) if valid else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_discovery_cache(path, root, files):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
            json.dump(
                {"version": _CACHE_VERSION, "root": str(root.resolve()), "files": files},
                handle,
                separators=(",", ":"),
            )
            temporary = handle.name
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except (OSError, UnboundLocalError):
            pass


def _discover_tasks(tasks_path=None, refresh=False, cache_path=None):
    """Discover module-level Task declarations, reusing unchanged per-file AST results."""
    root = Path(tasks_path or _TASKS_PATH)
    use_cache = tasks_path is None or cache_path is not None
    cache_path = Path(cache_path or _CACHE_PATH)
    cached = {} if refresh or not use_cache else _load_discovery_cache(cache_path, root)
    records = {}

    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if (path.name.startswith("_") or "deprecated" in relative.parts
                or any(part.startswith((".", "_")) for part in relative.parts[:-1])):
            continue
        stat = path.stat()
        key = relative.as_posix()
        old = cached.get(key, {})
        tasks = old.get("tasks") if (
            old.get("mtime_ns") == stat.st_mtime_ns and old.get("size") == stat.st_size
        ) else _parse_task_file(path, relative)
        records[key] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "tasks": tasks}

    if use_cache:
        _save_discovery_cache(cache_path, root, records)

    maps = ({}, {})
    origins = ({}, {})
    for key, record in records.items():
        module_name = ".".join(Path(key).with_suffix("").parts)
        for name, class_name, kind in record["tasks"]:
            index = 1 if kind == "dev" else 0
            location = f"tasks/{key}:{class_name}"
            if name in maps[index]:
                raise RuntimeError(f"Duplicate task {name!r}: {origins[index][name]} and {location}")
            maps[index][name] = (module_name, class_name)
            origins[index][name] = location
    return maps


@functools.cache
def _task_maps():
    return _discover_tasks()


class _LazyMap(Mapping):
    def __init__(self, index, proxies=False):
        self.index = index
        self.proxies = proxies
        self._proxies = {}

    @property
    def _mapping(self):
        return _task_maps()[self.index]

    def __getitem__(self, key):
        if not self.proxies:
            return self._mapping[key]
        module_name, class_name = self._mapping[key]
        return self._proxies.setdefault(
            key, _PrettyLazy(key, module_name, class_name if self.index else None)
        )

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)


def _lazy_loader(task_name, module_name, class_name=None):
    module = importlib.import_module(f".tasks.{module_name}", _PACKAGE_NAME)
    if class_name is not None and task_name not in _REGISTRY:
        return getattr(module, class_name)
    return _REGISTRY[task_name]


class _PrettyLazy:
    def __init__(self, name, module_name, class_name=None):
        self.name = name
        self.module_name = module_name
        self.class_name = class_name
        self._obj = None

    @property
    def _resolved(self):
        if self._obj is None:
            self._obj = _lazy_loader(self.name, self.module_name, self.class_name)
        return self._obj

    def __getattr__(self, attr):
        return getattr(self._resolved, attr)

    def __call__(self, *args, **kwargs):
        return self._resolved(*args, **kwargs)

    def __repr__(self):
        return f"<lazy:{self.name}>"


_task_to_module_map = _LazyMap(0)
_dev_task_to_module_map = _LazyMap(1)
DATASETS = _LazyMap(0, proxies=True)
DEV_DATASETS = _LazyMap(1, proxies=True)


class SelfMock:
    def __getattribute__(self, name):
        raise RuntimeError("score_answer should not use self.")


def match_task_name(name, include_dev=False):
    names = list(DATASETS)
    if include_dev:
        names += list(DEV_DATASETS)
    names += list(COLLECTIONS)
    normalized = str(name).replace("_", "").lower()
    matches = [candidate for candidate in names if candidate.replace("_", "").lower() == normalized]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous task name {name!r}: {matches}")
    suggestions = get_close_matches(str(name), names, n=3)
    hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(f"Unknown task {name!r}.{hint}")


def get_task(name, *args, **kwargs):
    name = match_task_name(name, include_dev=True)
    if name in COLLECTIONS:
        module_name, class_name = COLLECTIONS[name]
        module = importlib.import_module(f".tasks.{module_name}", _PACKAGE_NAME)
        return getattr(module, class_name)(*args, **kwargs)
    catalog = DATASETS if name in DATASETS else DEV_DATASETS
    return catalog[name](*args, **kwargs)


def list_tasks(include_mutated=False, include_generated=False):
    """Return the shipped roster; optionally include experimental task families."""
    return [
        name for name, (module_name, _) in _task_to_module_map.items()
        if name not in IGNORED
        and (include_mutated or module_name.split(".", 1)[0] != "mutated")
        and (include_generated or module_name.split(".", 1)[0] != "generated")
    ]


def task_catalog(query="", include_generated=False, include_mutated=False, include_dev=False):
    """Return searchable task metadata without importing task modules.

    Status describes registry kind; origin describes source placement, not quality.
    Defaults match list_tasks(). Collection adapters are not individual tasks.
    """
    names = set(list_tasks(include_mutated, include_generated))
    declarations = [(name, module, cls, "active")
                    for name, (module, cls) in _task_to_module_map.items() if name in names]
    if include_dev:
        declarations += [(name, module, cls, "dev")
                         for name, (module, cls) in _dev_task_to_module_map.items()]
    parsed = {}
    rows = []
    terms = query.casefold().split()
    for name, module, cls, status in sorted(declarations):
        path = _TASKS_PATH.joinpath(*module.split(".")).with_suffix(".py")
        if path not in parsed:
            parsed[path] = {node.name: node for node in ast.parse(path.read_text()).body
                            if isinstance(node, ast.ClassDef)}
        node = parsed[path][cls]
        summary = ""
        for field in node.body:
            if (isinstance(field, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "summary"
                            for target in field.targets)
                    and isinstance(field.value, ast.Constant)
                    and isinstance(field.value.value, str)):
                summary = field.value.value
        origin = module.split(".")[0] if module.startswith(("generated.", "mutated.")) else "core"
        row = dict(name=name, summary=summary, status=status, origin=origin,
                   source=f"reasoning_core/tasks/{path.relative_to(_TASKS_PATH).as_posix()}",
                   line=node.lineno)
        haystack = " ".join(str(value) for value in row.values()).casefold()
        if all(term in haystack for term in terms):
            rows.append(row)
    return rows


def get_score_answer_fn(task_name, *args, **kwargs):
    task_name = match_task_name(task_name)
    if task_name in COLLECTIONS:
        return get_task(task_name).score_answer
    scorer = DATASETS[task_name].score_answer
    return lambda answer, entry: scorer(SelfMock(), answer, entry)


def score_answer(answer, entry):
    if isinstance(entry.metadata, str):
        entry = edict(dict(entry))
        entry.metadata = json.loads(entry.metadata)
    task_name = (entry.get("metadata", {}).get("_task") or entry.get("task")
                 or entry.get("metadata", {}).get("task"))
    if task_name == "rg":
        try:
            from reasoning_gym import get_score_answer_fn as reasoning_gym_scorer
        except ImportError:
            raise RuntimeError(
                "reasoning_gym is not installed; install it with: pip install reasoning_gym"
            )
        return reasoning_gym_scorer(entry["metadata"]["source_dataset"])(answer, entry)
    return get_score_answer_fn(task_name)(answer, entry)


def register_to_reasoning_gym():
    import reasoning_gym
    for task_name, task_cls_proxy in DATASETS.items():
        task = task_cls_proxy()
        if task_name not in reasoning_gym.factory.DATASETS:
            reasoning_gym.register_dataset(task_name, task.__class__, task.config.__class__)
