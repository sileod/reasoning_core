"""Single-step OpenEnv environment backed by reasoning-core scorers."""

from __future__ import annotations

import json
import os
import re
from itertools import islice
from typing import Any
from uuid import uuid4

from easydict import EasyDict as edict
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from reasoning_core import list_tasks, score_answer

try:
    from ..models import ReasoningCoreAction, ReasoningCoreObservation
except ImportError:
    from models import ReasoningCoreAction, ReasoningCoreObservation


DEFAULT_DATASET = os.getenv(
    "RC_HF_DATASET",
    "reasoning-core/formal-reasoning-env",
)
DEFAULT_SIZE = int(os.getenv("RC_DATASET_SIZE", "1000"))
DEFAULT_SEED = int(os.getenv("RC_SEED", "42"))
AVAILABLE_TASKS = frozenset(list_tasks())
XML_ANSWER_PATTERN = re.compile(
    r"<answer>(.*?)</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_metadata": value}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _task_name(entry: dict[str, Any]) -> str | None:
    metadata = _metadata_dict(entry.get("metadata"))
    value = metadata.get("_task") or entry.get("task") or metadata.get("task")
    return str(value) if value is not None else None


def _normalize_entry(entry: dict[str, Any], index: int) -> dict[str, Any] | None:
    task_name = _task_name(entry)
    if task_name not in AVAILABLE_TASKS:
        return None
    return {
        "id": str(entry.get("id", index)),
        "prompt": str(entry["prompt"]),
        "answer": str(entry["answer"]),
        "metadata": {"task": task_name, **_metadata_dict(entry.get("metadata"))},
    }


def _load_hub_entries(
    dataset_name: str,
    split: str,
    seed: int,
    size: int,
) -> list[dict[str, Any]]:
    from datasets import get_dataset_split_names, load_dataset

    split_names = get_dataset_split_names(dataset_name)
    source_split = split
    if source_split not in split_names:
        source_split = next(
            (name for name in ("test", "validation", "eval", "dev") if name in split_names),
            "train",
        )

    stream = load_dataset(dataset_name, split=source_split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=max(size * 4, 1000))
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(islice(stream, size * 4)):
        normalized = _normalize_entry(dict(row), index)
        if normalized is not None:
            entries.append(normalized)
        if len(entries) >= size:
            break
    if not entries:
        raise RuntimeError(f"No supported tasks found in {dataset_name}:{source_split}")
    return entries


def _extract_answer(answer: str) -> str:
    match = XML_ANSWER_PATTERN.search(answer)
    return match.group(1).strip() if match else answer.strip()


class ReasoningCoreEnvironment(Environment):
    """Formally scored symbolic reasoning tasks from reasoning-core."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._entries: list[dict[str, Any]] = []
        self._entry_index = 0
        self._current_entry: dict[str, Any] | None = None
        self._configuration: tuple[str, str, int, int] | None = None

    def _configure(
        self,
        dataset_name: str,
        split: str,
        seed: int,
        size: int,
    ) -> None:
        configuration = (dataset_name, split, seed, size)
        if configuration == self._configuration:
            return
        self._entries = _load_hub_entries(dataset_name, split, seed, size)
        self._configuration = configuration
        self._entry_index = 0

    def reset(
        self,
        dataset_name: str = DEFAULT_DATASET,
        split: str = "train",
        seed: int = DEFAULT_SEED,
        size: int = DEFAULT_SIZE,
        episode_id: str | None = None,
    ) -> ReasoningCoreObservation:
        if size <= 0:
            raise ValueError("size must be positive")
        self._configure(dataset_name, split, seed, size)
        self._current_entry = self._entries[self._entry_index % len(self._entries)]
        self._entry_index += 1
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return ReasoningCoreObservation(
            prompt=self._current_entry["prompt"],
            score=None,
            correct_answer=None,
            task_name=_task_name(self._current_entry),
            dataset_metadata=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: ReasoningCoreAction) -> ReasoningCoreObservation:
        self._state.step_count += 1
        if self._current_entry is None:
            raise RuntimeError("Call reset() before step().")

        entry = edict(
            answer=self._current_entry["answer"],
            metadata=self._current_entry["metadata"],
        )
        score = float(score_answer(_extract_answer(action.answer), entry))
        return ReasoningCoreObservation(
            prompt=None,
            score=score,
            correct_answer=self._current_entry["answer"],
            task_name=_task_name(self._current_entry),
            dataset_metadata=self._current_entry["metadata"],
            done=True,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state
