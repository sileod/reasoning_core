from template import Task, Problem, Config
from easydict import EasyDict as edict
import random

class AdditionConfig(Config):
    max_value: int = 10

    def update(self, c):
        self.max_value += c * 5

class AdditionTask(Task):
    def generate(self):
        max_v = self.config.max_value

        a = random.randint(1, max_v)
        b = random.randint(1, max_v)

        return Problem(
            metadata=edict({
                "a": a,
                "b": b
            }),
            answer=str(a + b)
        )

    def prompt(self, metadata):
        return f"What is {metadata.a} + {metadata.b}?"