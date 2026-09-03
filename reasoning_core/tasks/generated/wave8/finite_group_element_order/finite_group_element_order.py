import random
from dataclasses import dataclass
from math import gcd

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'finite_group_element_order (draw 1 of 2)',
 'hypothesis': 'W1-020',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/finite_group_element_order',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 990936709,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def parse_order(answer):
    try:
        return int(str(answer).strip())
    except Exception:
        return None


@dataclass
class FiniteGroupElementOrderConfig(Config):
    n: int = 6

    def apply_difficulty(self, level):
        self.n = sround(self.n + level * 2)


class FiniteGroupElementOrder(Task):
    summary = "Given a Cayley table of a finite cyclic additive group Z_n and one of its elements, output that element's order under repeated group operation."
    task_version = 2
    config_cls = FiniteGroupElementOrderConfig

    def generate_entry(self):
        n = self.config.n
        while n < 2:
            n = self.config.n
        # distinct orders realizable in Z_n are the divisors of n
        divisors = sorted({n // gcd(n, g) for g in range(1, n + 1)})
        order = random.choice(divisors)
        # elements of this order: those g with gcd(n, g) == n // order
        wanted_gcd = n // order
        candidates = [g for g in range(n) if gcd(n, g) == wanted_gcd]
        g = random.choice(candidates)
        assert order >= 1
        assert (order * g) % n == 0
        for k in range(1, order):
            assert (k * g) % n != 0

        rows = []
        for i in range(n):
            rows.append(" ".join(str((i + j) % n) for j in range(n)))

        cayley_str = "\n".join(rows)
        metadata = edict({
            "n": int(n),
            "element": int(g),
            "order": int(order),
            "cayley": cayley_str,
        })
        metadata.payload = {
            "cayley": cayley_str,
        }

        return Entry(metadata=metadata, answer=str(int(order)))

    def render_prompt(self, metadata):
        lines = metadata.cayley.split("\n")
        header = "   " + " ".join(str(i) for i in range(metadata.n))
        body = "\n".join(f"{i}: {row}" for i, row in enumerate(lines))
        return (
            f"Below is the Cayley table of the group Z_{metadata.n}, where the row and column "
            f"labels are the group elements and each entry is the group product (mod {metadata.n}) "
            f"of the row element and the column element.\n"
            f"{header}\n{body}\n"
            f"Consider the element {metadata.element}. Its order is the smallest positive integer k "
            f"such that the element combined with itself k times equals the identity element 0.\n"
            f"The answer is the order, a single integer."
        )

    def score_answer(self, answer, entry):
        gold = parse_order(entry.answer)
        got = parse_order(answer)
        if got is None:
            return 0.0
        return 1.0 if got == gold else 0.0
