import os
import random
import re
import json

import numpy as np
from easydict import EasyDict as edict
from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)
from reasoning_core import get_task, list_tasks, score_answer


DEFAULT_SPLIT_SIZES = {
    "train": int(os.getenv("RC_NUM_TRAIN", "500")),
    "test": int(os.getenv("RC_NUM_TEST", "50")),
}
DEFAULT_SEED = int(os.getenv("RC_SEED", "0"))
DEFAULT_TASKS = sorted(list_tasks())
DEFAULT_PASS_THRESHOLD = float(os.getenv("RC_PASS_THRESHOLD", "0.9"))
XML_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
HF_DATASET_NAME = os.getenv("RC_HF_DATASET", "reasoning-core/symbolic-reasoning-env")
HF_DATASET_CONFIG = os.getenv("RC_HF_CONFIG")
DISABLE_HF_FALLBACK = os.getenv("RC_DISABLE_HF_FALLBACK", "0") == "1"


class ReasoningCoreTaskSpec(BaseModel):
    id: str
    prompt: str
    answer: str
    metadata: dict = Field(default_factory=dict)


class AnswerParams(BaseModel):
    answer: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _extract_final_answer(raw_answer: str) -> str:
    match = XML_ANSWER_PATTERN.search(raw_answer)
    if match:
        return match.group(1).strip()
    return raw_answer.strip()


def _generate_task_example(task_name: str, idx: int, seed: int) -> dict:
    _set_seed(seed + idx)
    example = get_task(task_name).generate_example()
    return {
        "id": f"{task_name}-{idx}",
        "prompt": example.prompt,
        "answer": str(example.answer),
        "metadata": dict(example.metadata),
    }


def _build_split(split_name: str, split_size: int, seed: int) -> list[dict]:
    if split_size <= 0:
        return []

    tasks = DEFAULT_TASKS
    task_count = len(tasks)
    examples: list[dict] = []
    for idx in range(split_size):
        task_name = tasks[idx % task_count]
        examples.append(_generate_task_example(task_name=task_name, idx=idx, seed=seed))
    return examples


def _parse_metadata(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {"raw_metadata": value}
    return {}


def _normalize_rows(rows: list[dict], prefix: str) -> list[dict]:
    normalized: list[dict] = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer", "")).strip()
        task_name = str(row.get("task", "task")).strip() or "task"
        metadata = _parse_metadata(row.get("metadata", {}))
        sample_id = str(row.get("id", f"{prefix}-{idx}"))
        normalized.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "answer": answer,
                "metadata": {"task": task_name, **metadata},
            }
        )
    return normalized


def _load_hf_tasks() -> tuple[list[dict], list[dict]] | None:
    if DISABLE_HF_FALLBACK:
        return None

    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"Could not import datasets for Hugging Face loading: {exc}")
        return None

    try:
        kwargs = {"path": HF_DATASET_NAME}
        if HF_DATASET_CONFIG:
            kwargs["name"] = HF_DATASET_CONFIG
        dataset = load_dataset(**kwargs)
    except Exception as exc:
        print(f"Could not load Hugging Face dataset '{HF_DATASET_NAME}': {exc}")
        return None

    if "train" not in dataset or "test" not in dataset:
        print(f"Dataset '{HF_DATASET_NAME}' does not expose both train and test splits.")
        return None

    train_rows = [dict(row) for row in dataset["train"]]
    test_rows = [dict(row) for row in dataset["test"]]
    train_tasks = _normalize_rows(train_rows, "train")
    test_tasks = _normalize_rows(test_rows, "test")
    print(
        "Loaded tasks from Hugging Face dataset "
        f"'{HF_DATASET_NAME}' (train={len(train_tasks)}, test={len(test_tasks)})."
    )
    return train_tasks, test_tasks


def _build_tasks() -> tuple[list[dict], list[dict]]:
    hf_tasks = _load_hf_tasks()
    if hf_tasks is not None:
        return hf_tasks

    print("Falling back to procedural task generation.")
    train_tasks = _build_split("train", DEFAULT_SPLIT_SIZES["train"], DEFAULT_SEED)
    test_tasks = _build_split("test", DEFAULT_SPLIT_SIZES["test"], DEFAULT_SEED + 10_000)
    return train_tasks, test_tasks


TRAIN_TASKS, TEST_TASKS = _build_tasks()


class ReasoningCore(Environment):
    """OpenReward environment for reasoning-core procedural tasks."""

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = ReasoningCoreTaskSpec.model_validate(task_spec)

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return TRAIN_TASKS
        if split == "test":
            return TEST_TASKS
        raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    def get_prompt(self):
        return [TextBlock(type="text", text=self.config.prompt)]

    @tool
    def answer(self, params: AnswerParams) -> ToolOutput:
        """Submit your final answer. This ends the episode."""
        completion = _extract_final_answer(params.answer)
        entry = edict(answer=self.config.answer, metadata=self.config.metadata)
        reward = float(score_answer(completion, entry))
        rounded_reward = f"{reward:.3f}"

        if reward >= DEFAULT_PASS_THRESHOLD:
            agent_message = f"Accepted (reward={rounded_reward})"
        elif reward > 0:
            agent_message = f"Partially correct (reward={rounded_reward})"
        else:
            agent_message = f"Incorrect (reward={rounded_reward})"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=agent_message)],
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    Server([ReasoningCore]).run()
