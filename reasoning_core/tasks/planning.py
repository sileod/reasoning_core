"""Cue-unique grounded STRIPS planning."""
from dataclasses import dataclass

from reasoning_core.tasks.planning_constructive import generate, render, score
from reasoning_core.template import Config, Task


@dataclass
class PlanningConfig(Config):
    horizon: int = 3

    def apply_difficulty(self, level):
        self.horizon += level


class Planning(Task):
    summary = "Find the unique cue-constrained plan that achieves a stated goal."
    task_name = "planning"
    task_version = 3
    config_cls = PlanningConfig

    def generate_entry(self):
        return generate(self.config.level, self.config.horizon)

    def render_prompt(self, metadata):
        return render(metadata)

    def score_answer(self, answer, entry):
        return score(answer, entry)
