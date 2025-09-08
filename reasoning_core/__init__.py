# __init__.py

import importlib
import pkgutil
import ast
from itertools import islice, cycle
from math import ceil
from tqdm.auto import tqdm
import os
from lazy_object_proxy import Proxy
from .tasks import _reasoning_gym
from .template import _REGISTRY
from . import tasks

class _PrettyLazy:
    __slots__ = ("name", "_p")

    def __init__(self, name, module_name):
        self.name = name
        self._p = Proxy(lambda task=name, module=module_name: _lazy_loader(task, module))

    def __getattr__(self, attr):
        return getattr(self._p, attr)

    def __call__(self, *args, **kwargs):
        return self._p(*args, **kwargs)

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
                try:
                    tree = ast.parse(f.read(), filename=filename)
                except SyntaxError:
                    continue  # Skip files with syntax errors

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and any(b.id == 'Task' for b in node.bases if isinstance(b, ast.Name)):
                    # Default task_name is the class name in lowercase
                    task_name = node.name.lower()
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
    # This import will trigger the __init_subclass__ for all tasks in the file,
    # populating _REGISTRY.
    importlib.import_module(f".tasks.{module_name}", __package__)
    return _REGISTRY[task_name]

_task_to_module_map = _discover_tasks()

DATASETS = {
    task_name: _PrettyLazy(task_name, module_name)
    for task_name, module_name in _task_to_module_map.items()
}

def get_task(k):
    return DATASETS[k]

def list_tasks():
    return list(DATASETS.keys())

class SelfMock:
    def __getattribute__(self,_): raise RuntimeError("score_answer should not use self.")



scorers = {
    k: lambda answer, entry, task_name=k: DATASETS[task_name].score_answer(SelfMock(), answer, entry)
    for k in DATASETS.keys()
}


scorers['RG'] = _reasoning_gym.RG().score_answer

def get_score_answer_fn(task_name, *args, **kwargs):
    if task_name in scorers:
        return scorers[task_name]
    raise ValueError(f"Task {task_name} not found. Available: {list(DATASETS.keys())}")

def score_answer(answer, entry):
    task_name = entry.get('metadata', {}).get('task', None) or entry.get('task', None)
    if task_name in scorers:
        return scorers[task_name](answer, entry)
    raise ValueError(f"Task {task_name} not found in entry. Available: {list(DATASETS.keys())}")

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