"""Ellipsis completion: recover omitted predicate arguments from dialogue.

A dialogue records a sequence of small arithmetic transactions between a
customer and a cashier. A later exchange is elliptical: the cashier's reply
omits its computed amount ("that was <blank>"), and the amount must be
recovered by resolving the ellipsis against the referenced transactions and
applying the stated operation on that subset.
"""

import random

from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload


@dataclass
class EllipsisConfig(Config):
    n_turns: int = 3
    n_digits: int = 2

    def apply_difficulty(self, level):
        self.n_turns = int(3 + 2 * (level // 2))
        self.n_digits = int(2 + (level // 3))


ACTIONS = ["paid", "handed over", "settled", "covered", "rang up"]
OBJECTS = ["the bill", "the order", "the tab", "the total"]
MEASURES = ["dollars", "euros", "pounds"]
SUB_Q = [
    "the first", "the second", "the third", "the fourth", "the fifth",
    "the sixth",
]
SUBSET_WORDS = [
    "put together", "combined", "added up",
    "in total", "altogether",
]


def _amount(n_digits):
    lo = 10 ** (n_digits - 1)
    hi = 10 ** n_digits - 1
    return random.randint(lo, hi)


def _gen_amounts(n_digits, n):
    return [_amount(n_digits) for _ in range(n)]


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class EllipsisCompletion(Task):
    summary = (
        "Complete an elliptical reply by recovering omitted predicates and "
        "arguments from preceding dialogue, resolving the combined amount of a "
        "referenced transaction subset under varied action/object/measure "
        "wording, subset size, and dialogue depth."
    )
    config_cls = EllipsisConfig

    def generate_entry(self):
        c = self.config
        n_turns = c.n_turns
        n_digits = c.n_digits

        amounts = _gen_amounts(n_digits, n_turns)
        lines = []
        for i in range(n_turns):
            action = random.choice(ACTIONS)
            obj = random.choice(OBJECTS)
            measure = random.choice(MEASURES)
            lines.append(
                "Customer {1} {0} {2} for {3} {4}.".format(
                    action, i + 1, obj, amounts[i], measure
                )
            )

        # Choose a subset of 2 or 3 distinct transaction indices to combine.
        k = random.choice([2, 3])
        idxs = sorted(random.sample(range(n_turns), k))
        total = sum(amounts[i] for i in idxs)

        subset_desc = _describe_subset(idxs, n_turns)
        combine = random.choice(SUBSET_WORDS)
        customer_q = (
            "Later the customer asked what the {0} {1} came to.".format(
                subset_desc, combine
            )
        )
        cashier = random.choice(FOLLOWUPS_ALL)

        dialogue = "\n".join(lines) + "\n" + customer_q + "\n" + (
            "The cashier replied, elliptically: \"{0} <blank>.\"".format(cashier)
        )

        metadata = edict({
            "dialogue": dialogue,
            "cashier": cashier,
            "target_indices": idxs,
            "amounts": amounts,
            "total": total,
        })
        metadata.payload = {
            "dialogue": dialogue,
            "cashier": cashier,
            "target_indices": idxs,
            "amounts": amounts,
        }
        return Entry(metadata=metadata, answer=str(total))

    def render_prompt(self, metadata):
        payload = metadata.payload
        dialogue = payload["dialogue"]
        return (
            render_payload({"dialogue": dialogue})
            + "\n\n"
            + "The cashier's elliptical reply \""
            + payload["cashier"]
            + " <blank>\" is missing its amount. Recover the amount by "
            + "resolving the ellipsis: the missing figure is the sum of the "
            + "amounts of the transactions the customer's question referenced "
            + "("
            + ", ".join(_ordinal(i + 1) for i in payload["target_indices"])
            + "). What integer amount does the ellipsis resolve to?\n\n"
            + "The answer is a single integer."
        )

    def score_answer(self, answer, entry):
        try:
            val = int(str(answer).strip())
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if val == int(entry.answer) else 0.0


FOLLOWUPS_ALL = [
    "that was", "it came to", "that ran to", "that totalled",
    "that added up to", "the amount was",
]


def _describe_subset(idxs, n_turns):
    parts = [_ordinal(i + 1) + " transaction" for i in idxs]
    if len(parts) == 2:
        return parts[0] + " and " + parts[1]
    return parts[0] + ", " + parts[1] + ", and " + parts[2]


TASK_META = {'parent_source_id': None,
 'idea': 'ellipsis_completion (draw 1 of 2)',
 'hypothesis': 'ASTRA0-12',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/ellipsis_completion',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1777646155,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
