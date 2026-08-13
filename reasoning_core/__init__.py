# __init__.py


__version__ = "0.5.0"

import importlib
#import pkgutil
import ast
from itertools import islice, cycle
from math import ceil
import json
from tqdm.auto import tqdm
from pathlib import Path
from .template import _REGISTRY, edict, prepr_task_name
from . import tasks
from .zero_shot_eval import evaluate_model

_PACKAGE_NAME = __name__ 

COLLECTIONS = {
    "procedural_warmup": ("_procedural_warmup", "ProceduralWarmup"),
    "reasoning_gym": ("_reasoning_gym", "Reasoning_Gym"),
    "synlogic": ("_synlogic", "Synlogic"),
}


class _PrettyLazy:
    def __init__(self, name, module_name):
        self.name = name
        self.module_name = module_name
        self._obj = None

    @property
    def _resolved(self):
        if self._obj is None:
            self._obj = _lazy_loader(self.name, self.module_name)
        return self._obj

    def __getattr__(self, attr):
        return getattr(self._resolved, attr)

    def __call__(self, *args, **kwargs):
        return self._resolved(*args, **kwargs)

    def __repr__(self):
        return f"<lazy:{self.name}>"

def _discover_tasks(tasks_path=None):
    """
    Parses task files to find all Task subclasses and their names without importing them.
    Returns a mapping of {task_name: module_name}.
    """
    task_map = {}
    dev_task_map = {}
    tasks_path = Path(tasks_path or tasks.__path__[0])
    for path in sorted(tasks_path.rglob("*.py")):
        relative = path.relative_to(tasks_path)
        if (path.name.startswith("_") or "deprecated" in relative.parts
                or any(part.startswith((".", "_")) for part in relative.parts[:-1])):
            continue
        module_name = ".".join(relative.with_suffix("").parts)
        tree = ast.parse(path.read_text(), filename=str(relative))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                b.id if isinstance(b, ast.Name) else b.attr
                for b in node.bases
                if isinstance(b, (ast.Name, ast.Attribute))
            }
            if not {'Task', 'DevTask'} & bases:
                continue
            task_name = prepr_task_name(node.name)
            for body_item in node.body:
                if (isinstance(body_item, ast.Assign) and
                    len(body_item.targets) == 1 and
                    isinstance(body_item.targets[0], ast.Name) and
                    body_item.targets[0].id == 'task_name' and
                    isinstance(body_item.value, ast.Constant) and
                    isinstance(body_item.value.value, str)):
                    task_name = body_item.value.value
                    break
            target_map = dev_task_map if 'DevTask' in bases else task_map
            target_map[task_name] = (module_name, node.name)
    return task_map, dev_task_map


def _lazy_loader(task_name, module_name, class_name=None):
    """Triggers the module import and returns the specific task class from the registry."""
    module = importlib.import_module(f".tasks.{module_name}", _PACKAGE_NAME)
    if class_name is not None and task_name not in _REGISTRY:
        return getattr(module, class_name)
    return _REGISTRY[task_name]

_task_to_module_map, _dev_task_to_module_map = _discover_tasks()

DATASETS = {
    task_name: _PrettyLazy(task_name, module_name)
    for task_name, (module_name, class_name) in _task_to_module_map.items()
}

DEV_DATASETS = {
    task_name: _PrettyLazy(task_name, module_name)
    for task_name, (module_name, class_name) in _dev_task_to_module_map.items()
}

class SelfMock:
    def __getattribute__(self,_): raise RuntimeError("score_answer should not use self.")



scorers = {
    k: lambda answer, entry, task_name=k: DATASETS[task_name].score_answer(SelfMock(), answer, entry)
    for k in DATASETS.keys()
}


for _collection in COLLECTIONS:
    scorers[_collection] = lambda a, e, name=_collection: get_task(name).score_answer(a, e)

def match_task_name(name, include_dev=False):
    datasets = list(DATASETS.keys())
    if include_dev:
        datasets += list(DEV_DATASETS.keys())
    datasets += list(COLLECTIONS)
    norm = lambda x: x.replace('_','').lower()
    matches = [t for t in datasets if norm(name)==norm(t)]
    assert len(matches)==1, f"Could not uniquely identify task {name} in {datasets}"
    return matches[0]

def get_task(k, *args, **kwargs):
    k=match_task_name(k, include_dev=True)
    if k in COLLECTIONS:
        module_name, class_name = COLLECTIONS[k]
        module = importlib.import_module(f".tasks.{module_name}", _PACKAGE_NAME)
        return getattr(module, class_name)(*args, **kwargs)
    if k in DATASETS:
        return DATASETS[k](*args, **kwargs)
    module_name, class_name = _dev_task_to_module_map[k]
    cls = _lazy_loader(k, module_name, class_name)
    return cls(*args, **kwargs)

DEPRECATED = ['symbolic_arithmetics', 'graph_node_centrality']
# count_elements absorbed into set_expression's multiset Count(x, S) mode (toyish standalone; zero-shot 1.0)
ignored = DEPRECATED + ['reasonining_gym', 'count_elements']

def list_tasks():
    return [k for k in DATASETS.keys() if k not in ignored]


def get_score_answer_fn(task_name, *args, **kwargs):
    task_name = match_task_name(task_name)
    return scorers[task_name]
    

def score_answer(answer, entry):
    if isinstance(entry.metadata, str):
        entry = edict(dict(entry))
        entry.metadata = json.loads(entry.metadata)
    task_name = entry.get('metadata', {}).get('_task', None) or entry.get('task', None) or entry.get('metadata', {}).get('task', None)

    if task_name=="rg":
        try:
            from reasoning_gym import get_score_answer_fn
        except ImportError:
            raise RuntimeError("reasoning_gym is not installed; install it with: pip install reasoning_gym")
        scorer = get_score_answer_fn(entry['metadata']['source_dataset'])
        return scorer(answer, entry)

    task_name= match_task_name(task_name)
    return scorers[task_name](answer, entry)

def generate_dataset(num_samples=100, tasks=None, batch_size=4):
    tasks = list(tasks or list_tasks())
    n = ceil(num_samples / batch_size)
    batches = [get_task(t)().generate_balanced_batch(batch_size) 
               for t in tqdm(islice(cycle(tasks), n))]
    return [ex for b in batches for ex in b][:num_samples]

def register_to_reasoning_gym():
    import reasoning_gym
    for task_name, task_cls_proxy in DATASETS.items():
        # Accessing the proxy triggers the lazy load
        task = task_cls_proxy()
        if task_name not in reasoning_gym.factory.DATASETS:
            reasoning_gym.register_dataset(task_name, task.__class__, task.config.__class__)


__all__ = ["DATASETS", "get_score_answer_fn", "register_to_reasoning_gym"]
