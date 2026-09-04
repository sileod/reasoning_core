"""First-order formulas over explicit finite models: nested quantifiers, witness answer.

The formula has the shape  EX x. ALL y. EX z. ( (y,z) in R(x)  and  z != f[y] ),
where D = {0..n-1} is a finite domain, f is an explicit unary function table and
R(x) is an explicit finite relation on D^2 given per x.  The formula is built so
that a smallest witness x in D always exists; the canonical answer is that witness.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround
from reasoning_core.utils import score_scalar


@dataclass
class FmQuantConfig(Config):
    n: int = 5
    extra: int = 1

    def apply_difficulty(self, level):
        self.n = sround(5 + level)
        self.extra = sround(1 + level // 2)


def _partners(n, f, x, m, extra):
    """Build the ordered list of allowed (y,z) pairs in R(x).

    Guarantees: for x < m the row y=0 has only the partner (0, f[0]);
    for x >= m every y has a partner with z != f[y].  As a result the
    smallest x satisfying ALL y EX z ( (y,z) in R(x) and z != f[y] ) is m.
    """
    pairs = []
    f0 = f[0]
    for y in range(n):
        pairs.append((y, f[y]))
        if x < m:
            if y > 0:
                pairs.append((y, (f[y] + 1) % n))
        else:
            pairs.append((y, (f[y] + 1) % n))
    for _ in range(extra):
        y = random.randrange(n)
        if x < m and y == 0:
            z = f0
        else:
            z = random.randrange(n)
        pairs.append((y, z))
    return pairs


class FiniteModelQuantifiers(Task):
    summary = "Evaluate nested EXISTS/ALL/EXISTS first-order formulas over an explicit finite domain, a unary function table and a per-element binary relation, returning the smallest witnessing value or none."
    config_cls = FmQuantConfig

    def generate_entry(self):
        n = int(self.config.n)
        extra = int(self.config.extra)
        f = [random.randrange(n) for _ in range(n)]
        m = random.randrange(n)
        rel = {}
        for x in range(n):
            rel[x] = _partners(n, f, x, m, extra)
        answer = str(m)
        payload = {
            "domain": n,
            "function": list(f),
            "relation": [list(rel[_x]) for _x in range(n)],
        }
        for x in range(m):
            assert not any((0, zz) in rel[x] and zz != f[0] for zz in range(n)), (
                "row below witness must fail")
        assert all(any((y, zz) in rel[m] and zz != f[y] for zz in range(n)) for y in range(n)), (
            "witness row must satisfy the formula")
        metadata = {"payload": payload, "f": list(f), "m": int(m), "n": int(n)}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        n = metadata.payload["domain"]
        f = metadata.payload["function"]
        rel = metadata.payload["relation"]
        dset = ", ".join(str(i) for i in range(n))
        flines = ", ".join("f[" + str(y) + "]=" + str(f[y]) for y in range(n))
        rlines = []
        for x in range(n):
            pl = ", ".join("(" + str(y) + "," + str(z) + ")" for (y, z) in rel[x])
            rlines.append("for x=" + str(x) + ": {" + pl + "}")
        rtext = "; ".join(rlines)
        return (
            "We reason over the finite domain D = {" + dset + "}.\n"
            "A unary function f is given on D: " + flines + ".\n"
            "A ternary relation R on D is given by the allowed pairs (y,z) for each x: "
            + rtext + ".\n"
            "Consider the first-order formula EX x in D. ALL y in D. EX z in D. such that "
            "((y,z) is in R(x)) and (z != f[y]).\n"
            "Give the smallest x in D for which this formula is true. "
            "The answer is that integer x and nothing else."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)


TASK_META = {'parent_source_id': None,
 'idea': 'finite_model_quantifiers (draw 1 of 1)',
 'hypothesis': 'HV-054',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/finite_model_quantifiers',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2104874763,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
