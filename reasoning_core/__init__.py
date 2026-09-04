"""Public Reasoning Core API."""

__version__ = "0.5.0"

from .registry import (
    DATASETS,
    DEV_DATASETS,
    _dev_task_to_module_map,
    _discover_tasks,
    _task_to_module_map,
    generate_dataset,
    get_score_answer_fn,
    get_task,
    list_tasks,
    match_task_name,
    prepr_task_name,
    register_to_reasoning_gym,
    score_answer,
)
from .source_store import SourceStore


def __getattr__(name):
    if name == "evaluate_model":
        from .zero_shot_eval import evaluate_model
        return evaluate_model
    raise AttributeError(name)


__all__ = [
    "DATASETS",
    "DEV_DATASETS",
    "SourceStore",
    "generate_dataset",
    "get_score_answer_fn",
    "get_task",
    "list_tasks",
    "match_task_name",
    "register_to_reasoning_gym",
    "score_answer",
]
