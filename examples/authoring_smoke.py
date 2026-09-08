"""Starter: copy into reasoning_core/tasks/example_maximum.py, then validate via the CLI."""

import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, stochastic_rounding


@dataclass
class MaximumConfig(Config):
    count: int = 3

    def apply_difficulty(self, level):
        self.count = stochastic_rounding(self.count + level)


class ExampleMaximum(Task):
    summary = "Find the maximum integer in a randomly generated list of signed integers."
    config_cls = MaximumConfig

    def generate_entry(self):
        values = random.sample(range(-100, 101), self.config.count)
        return Entry(metadata={"values": values}, answer=str(max(values)))

    def render_prompt(self, metadata):
        return f"Find the maximum of {metadata['values']!r}. The answer is one integer."
