import os
import random
import re
from pathlib import Path

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


TRAIN_TASKS = _build_split("train", DEFAULT_SPLIT_SIZES["train"], DEFAULT_SEED)
TEST_TASKS = _build_split("test", DEFAULT_SPLIT_SIZES["test"], DEFAULT_SEED + 10_000)


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
    data_dir = Path(os.getenv("ORWD_DATA_DIR", "/orwd_data"))
    if data_dir.exists():
        print(f"Using mounted data directory: {data_dir}")
    Server([ReasoningCore]).run()
