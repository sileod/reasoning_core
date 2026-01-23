# __init__.py


__version__ = "0.1.17"

import importlib
#import pkgutil
import ast
import copy
from itertools import islice, cycle
from math import ceil
import json
from tqdm.auto import tqdm
import os
from .template import _REGISTRY, prepr_task_name
from . import tasks

_PACKAGE_NAME = __name__ 


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

def _discover_tasks():
    """
    Parses task files to find all Task subclasses and their names without importing them.
    Returns a mapping of {task_name: module_name}.
    """
    task_map = {}
    tasks_path = tasks.__path__[0]
    for filename in os.listdir(tasks_path):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            with open(os.path.join(tasks_path, filename), 'r') as f:
                tree = ast.parse(f.read(), filename=filename)


            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and any(b.id == 'Task' for b in node.bases if isinstance(b, ast.Name)):
                    # Default task_name is the class name in lowercase
                    task_name = prepr_task_name(node.name)
                    # Look for an explicit `task_name = "..."` assignment
                    for body_item in node.body:
                        if (isinstance(body_item, ast.Assign) and
                            len(body_item.targets) == 1 and
                            isinstance(body_item.targets[0], ast.Name) and
                            body_item.targets[0].id == 'task_name'):
                            # For Python 3.8+ value is Constant, for older it's Str
                            if isinstance(body_item.value, (ast.Constant, ast.Str)):
                                task_name = body_item.value.s
                            break
                    task_map[task_name] = module_name
    return task_map


def _lazy_loader(task_name, module_name):
    """Triggers the module import and returns the specific task class from the registry."""
    importlib.import_module(f".tasks.{module_name}", _PACKAGE_NAME)
    return _REGISTRY[task_name]

_task_to_module_map = _discover_tasks()

DATASETS = {
    task_name: _PrettyLazy(task_name, module_name)
    for task_name, module_name in _task_to_module_map.items()
}

class SelfMock:
    def __getattribute__(self,_): raise RuntimeError("score_answer should not use self.")



scorers = {
    k: lambda answer, entry, task_name=k: DATASETS[task_name].score_answer(SelfMock(), answer, entry)
    for k in DATASETS.keys()
}


def rg_scorer(a, e):
    from .tasks import _reasoning_gym
    return _reasoning_gym.RG().score_answer(a, e)

scorers['RG'] = lambda a, e: rg_scorer(a, e)

def match_task_name(name):
    norm = lambda x: x.replace('_','').lower()
    matches = [t for t in DATASETS.keys() if norm(name)==norm(t)]
    assert len(matches)==1, f"Could not uniquely identify task {name} in {list(DATASETS.keys())}"
    return matches[0]

def get_task(k, *args, **kwargs):
    k=match_task_name(k)
    return DATASETS[k](*args, **kwargs)

def list_tasks():
    return list(DATASETS.keys())


def get_score_answer_fn(task_name, *args, **kwargs):
    task_name = match_task_name(task_name)
    return scorers[task_name]
    

def score_answer(answer, entry):
    if type(entry.metadata)==str:
        entry=copy.deepcopy(entry)
        entry.metadata = json.loads(entry.metadata)
    task_name = entry.get('metadata', {}).get('_task', None) or entry.get('task', None) or entry.get('metadata', {}).get('task', None)

    if task_name=="rg":
        from reasoning_gym import get_score_answer_fn
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
