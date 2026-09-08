"""Public Reasoning Core API."""

__version__ = "0.5.0"

# Only the intended API is bound here. The registry's discovery internals
# (_discover_tasks, _task_to_module_map, _dev_task_to_module_map) stay in
# reasoning_core.registry: __all__ hides them from `import *`, but binding them at the package
# root still publishes reasoning_core._discover_tasks as something callers can reach for and
# then depend on. prepr_task_name is likewise left where its only caller already imports it.
from .registry import (
    DATASETS,
    DEV_DATASETS,
    get_score_answer_fn,
    get_task,
    list_tasks,
    match_task_name,
    register_to_reasoning_gym,
    score_answer,
    task_catalog,
)
from .source_store import SourceStore


def __getattr__(name):
    if name == "evaluate_model":
        from .evaluation.zero_shot import evaluate_model
        return evaluate_model
    raise AttributeError(name)


__all__ = [
    "DATASETS",
    "DEV_DATASETS",
    "SourceStore",
    "get_score_answer_fn",
    "get_task",
    "list_tasks",
    "match_task_name",
    "register_to_reasoning_gym",
    "score_answer",
    "task_catalog",
]
