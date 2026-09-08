import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


@dataclass
class IncrementalInterpretationConfig(Config):
    max_val: int = 8
    n_fragments: int = 5

    def apply_difficulty(self, level):
        self.max_val = 5 + 2 * level
        self.n_fragments = 3 + level


def _feasible(max_val, constraints):
    fe = set(range(1, max_val + 1))
    for kind, arg in constraints:
        if kind == "atleast":
            fe = {v for v in fe if v >= arg}
        elif kind == "atmost":
            fe = {v for v in fe if v <= arg}
        elif kind == "equals":
            fe = {v for v in fe if v == arg}
        else:
            fe = {v for v in fe if v != arg}
    return fe


def _render_constraint(c):
    kind, arg = c
    if kind == "atleast":
        return f"x is at least {arg}."
    if kind == "atmost":
        return f"x is at most {arg}."
    if kind == "equals":
        return f"x equals {arg}."
    return f"x is not {arg}."


class IncrementalInterpretation(Task):
    summary = "After each sentence fragment, report which interpretations remain possible using only the observed prefix."

    config_cls = IncrementalInterpretationConfig

    def generate_entry(self):
        cfg = self.config
        max_val = cfg.max_val
        n = cfg.n_fragments

        constraints = []
        prefix_feasible = []
        for _ in range(n):
            kind = random.choice(["atleast", "atmost", "not", "equals"])
            arg = random.randint(1, max_val)
            constraints.append((kind, arg))
            prefix_feasible.append(_feasible(max_val, constraints))

        nonempty = [r for r in range(n) if len(prefix_feasible[r]) > 0]
        if not nonempty:
            row = n - 1
            remaining = []
        else:
            row = random.choice(nonempty)
            remaining = sorted(prefix_feasible[row])

        shown = constraints[: row + 1]

        if len(remaining) == 0:
            answer = "0"
        else:
            answer = " ".join(str(v) for v in remaining)

        metadata = edict({
            "max_val": max_val,
            "constraints": shown,
            "index": row,
            "feasible": remaining,
        })
        metadata.payload = {
            "max": max_val,
            "fragments": " ".join(_render_constraint(c) for c in shown),
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"A hidden integer x satisfies 1 <= x <= {metadata.payload['max']}. "
            f"Facts about x are revealed one at a time. Only the facts shown below have been observed so far.\n\n"
            f"{metadata.payload['fragments']}\n\n"
            f"Considering ONLY these observed facts, list every integer in [1, {metadata.payload['max']}] that could "
            f"still be the true value of x, in ascending order, separated by spaces. If no integer is consistent with "
            f"the observed facts, write the single number 0.\n\n"
            f"Answer format: a space-separated list of integers in ascending order. Example: 1 3 5 or 0"
        )

    def score_answer(self, answer, entry):
        try:
            got = [int(v) for v in str(answer).strip().split()]
        except Exception:
            return 0.0
        gold = [int(v) for v in entry.answer.split()]
        if got == gold:
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'incremental_interpretation (draw 1 of 2)',
 'hypothesis': 'ASTRA2-incremental_interpretation',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/incremental_interpretation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 584479410,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
