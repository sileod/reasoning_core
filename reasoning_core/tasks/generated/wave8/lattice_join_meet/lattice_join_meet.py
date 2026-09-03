"""Given a finite lattice and two elements, output their join and meet."""

import math
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'lattice_join_meet (draw 1 of 2)',
 'hypothesis': 'W1-026',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/lattice_join_meet',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 991003721,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _decompose(n, rng):
    """Return a sorted list of rng-chosen proper divisors of n (each >= 2)."""
    divs = [d for d in range(2, n) if n % d == 0]
    rng.shuffle(divs)
    k = rng.randint(0, min(3, len(divs)))
    return sorted(divs[:k])


def _finish(n, divs, rng):
    """Extend divs (a decreasing chain of divisors) to a full chain from n down to 1."""
    divs = sorted(divs)
    chain = [n]
    cur = n
    for d in divs:
        if d < cur and cur % d == 0:
            chain.append(d)
            cur = d
    chain.append(1)
    return chain


def _divisors_of(n):
    return sorted(d for d in range(1, n + 1) if n % d == 0)


class LatticeJoinMeetConfig(Config):
    level: int = 0
    n_cap: int = 60

    def apply_difficulty(self, level):
        self.level = level
        self.n_cap = 12 + 10 * level


class LatticeJoinMeet(Task):
    summary = ("Given a finite divisibility (gcd-lattice) on the divisors of an integer, "
               "output the join (lcm of two elements) and meet (gcd of two elements).")

    config_cls = LatticeJoinMeetConfig

    def generate_entry(self):
        cfg = self.config
        n_cap = cfg.n_cap
        n = random.randint(16, max(16, n_cap))
        divs = _decompose(n, random)
        chain = _finish(n, divs, random)
        universe = _divisors_of(n)
        max_idx = len(universe) - 1
        # Choose two distinct elements in the lattice.
        while True:
            i = random.randrange(len(universe))
            j = random.randrange(len(universe))
            if i != j:
                break
        a = universe[i]
        b = universe[j]
        join = a * b // math.gcd(a, b)          # lcm, in lattice
        meet = math.gcd(a, b)                    # gcd
        # Both are divisors of n (lattice proper).
        assert n in universe and 1 in universe
        metadata = edict({
            "n": n,
            "a": a,
            "b": b,
            "join": join,
            "meet": meet,
            "chain": chain,
        })
        metadata.payload = {
            "given": f"a divisibility lattice on the divisors of {n}: "
                     f"the elements are {universe}, ordered by divisibility "
                     f"(x <= y when x divides y), so the join of x and y is "
                     f"their least common multiple and the meet is their greatest "
                     f"common divisor.",
            "prompt": f"Consider the two elements a = {a} and b = {b} "
                      f"of this lattice. What are their join and meet?",
        }
        answer = f"{join} {meet}"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return render_payload(metadata.payload) + (
            "\n\nThe answer is the two integers separated by a single space: "
            "the join first, then the meet."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        parts = answer.strip().split()
        if len(parts) != 2:
            return 0.0
        try:
            j = int(parts[0])
            m = int(parts[1])
        except ValueError:
            return 0.0
        if j == entry.metadata.join and m == entry.metadata.meet:
            return 1.0
        return 0.0
