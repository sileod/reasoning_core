"""Clarification question: ask the shortest question that distinguishes two requests needing different answers, or answer directly when already determined."""

import random
import re
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'clarification_question (draw 1 of 2)',
 'hypothesis': 'ASTRA0-05',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/clarification_question',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1817164334,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ClarificationQuestionConfig(Config):
    max_value: int = 10

    def apply_difficulty(self, level):
        self.max_value = sround(10 + 5 * level)


TARGET = {
    "the sum": ("the sum", lambda a, b: a + b),
    "the product": ("the product", lambda a, b: a * b),
    "the absolute difference": ("the absolute difference", lambda a, b: abs(a - b)),
    "the larger": ("the larger", lambda a, b: max(a, b)),
    "the smaller": ("the smaller", lambda a, b: min(a, b)),
}

# Pairs of targets that, interpreted as two readings, can disagree.
_CANDIDATES = [
    ("the sum", "the product"),
    ("the sum", "the absolute difference"),
    ("the product", "the absolute difference"),
    ("the larger", "the smaller"),
    ("the sum", "the larger"),
    ("the product", "the larger"),
    ("the absolute difference", "the smaller"),
]


def _apply(tn, x, y):
    _, f = TARGET[tn]
    return int(f(x, y))


class ClarificationQuestion(Task):
    summary = ("Ask the shortest question that disambiguates one ambiguous request whose two readings "
               "give different numbers; give both resulting numbers as the answer. Covers sums, "
               "products, absolute differences, and larger/smaller, derived from one given pair with "
               "the two results spread across a wide numeric range.")
    config_cls = ClarificationQuestionConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(64):
            x = random.randint(1, cfg.max_value)
            y = random.randint(1, cfg.max_value)
            t1, t2 = random.choice(_CANDIDATES)
            v1 = _apply(t1, x, y)
            v2 = _apply(t2, x, y)
            if v1 != v2:
                break
        else:
            raise RuntimeError("could not find two distinct readings")

        body = _describe_note(t1, t2, x, y)
        gold = f"{min(v1, v2)} {max(v1, v2)}"

        metadata = edict({
            "x": x, "y": y,
            "t1": t1, "t2": t2,
            "v1": v1, "v2": v2,
            "body": body,
            "answer": gold,
        })
        return Entry(metadata=metadata, answer=gold)

    def render_prompt(self, metadata):
        return _render_prompt(metadata.body)

    def score_answer(self, answer, entry):
        gold = entry.answer
        s = answer.strip()
        nums = re.findall(r"-?\d+", s)
        if len(nums) == 2:
            a = int(nums[0])
            b = int(nums[1])
            ga, gb = (int(g) for g in gold.split())
            if (a == ga and b == gb) or (a == gb and b == ga):
                return 1.0
        return 0.0


def _describe_note(t1, t2, x, y):
    label1, _ = TARGET[t1]
    label2, _ = TARGET[t2]
    return (
        f"A friend texts you for help and asks only for \"{label1} of {x} and {y}.\" "
        f"On reflection you realize the request is ambiguous: it could equally mean "
        f"\"{label2} of {x} and {y},\" and the two readings give different numbers."
    )


def _render_prompt(body):
    return (
        f"{body}\n\n"
        f"Ask the single question that resolves which reading is meant, then give the number that "
        f"follows from each reading. Write the answer as the two integer results, the smaller first, "
        f"separated by a space (for example: 5 7)."
    )
