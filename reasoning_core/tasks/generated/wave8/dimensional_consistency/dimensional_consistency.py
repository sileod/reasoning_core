"""Given dimension vectors for variables in an equation, find the unknown
variable's exponent vector that makes the equation dimensionally valid."""

import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

_DIM_LABELS = ["M", "L", "T"]


def _zero():
    return [0, 0, 0]


def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mul(a, p):
    return [a[0] * p, a[1] * p, a[2] * p]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _neg(a):
    return [-a[0], -a[1], -a[2]]


def _fmt_vec(v):
    return f"({v[0]},{v[1]},{v[2]})"


def _fmt_nz(v):
    # minimal ordered list of nonzero coords for assertions
    return tuple(x for x in v if x != 0)


def _fmt_dim(v):
    parts = []
    for d, p in zip(_DIM_LABELS, v):
        if p == 0:
            continue
        parts.append(d if p == 1 else f"{d}^{p}")
    return "*".join(parts) if parts else "dimensionless"


def _fmt_term(parts):
    return "*".join(var if p == 1 else f"{var}^{p}" for var, p in parts)


def _label(i):
    # A, B, ..., Z, AA, AB, ...
    s = ""
    i = i + 1
    while i > 0:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


@dataclass
class DimensionalConsistencyConfig(Config):
    n_left: int = 3
    max_dim: int = 3
    max_pow: int = 3

    def apply_difficulty(self, level):
        self.n_left = sround(self.n_left + level)
        self.max_dim = sround(self.max_dim + level // 2)
        self.max_pow = sround(self.max_pow + level // 3)


def parse_answer(answer, entry):
    if isinstance(answer, str):
        try:
            ans = ast.literal_eval(answer.strip())
        except Exception:
            return None
        if isinstance(ans, (list, tuple)) and len(ans) == 3:
            return tuple(int(c) for c in ans)
    return None


class DimensionalConsistency(Task):
    summary = "Given dimension vectors for variables in a sum-of-products equation over M,L,T, return the exponent triple of the unknown variable X (a numeric witness) that makes the equation dimensionally valid."
    config_cls = DimensionalConsistencyConfig

    def generate_entry(self):
        cfg = self.config
        r = max(1, cfg.max_dim)
        n_left = max(2, cfg.n_left)

        while True:
            # One common dimension C shared by every term.
            C = [random.randint(-r, r) for _ in range(3)]
            if all(c == 0 for c in C):
                continue

            dims = {}  # var -> dim vector
            counter = [0]

            def newvar():
                name = _label(counter[0])
                counter[0] += 1
                return name

            left_terms = []
            for _ in range(n_left):
                if random.random() < 0.45:
                    # single-variable term: var with power 1 whose dim = C
                    var = newvar()
                    dims[var] = list(C)
                    coeff = random.randint(2, 5)
                    left_terms.append((coeff, [(var, 1)]))
                else:
                    # two-variable term: free factor V1^p times carrier V2 (power 1)
                    v1 = newvar()
                    p = random.randint(1, cfg.max_pow)
                    d1 = [random.randint(-r, r) for _ in range(3)]
                    dims[v1] = d1
                    v2 = newvar()
                    dims[v2] = _sub(C, _mul(d1, p))  # carrier makes term equal C
                    coeff = random.randint(2, 5)
                    left_terms.append((coeff, [(v1, p), (v2, 1)]))

            # Right side: one free factor R1 (power 1) times X (power 1).
            r1 = newvar()
            dr = [random.randint(-r, r) for _ in range(3)]
            dims[r1] = dr
            Xdim = _sub(C, dr)

            if all(x == 0 for x in Xdim):
                continue
            if any(abs(x) > r * 4 for x in Xdim):
                continue

            right_term = [(r1, 1), ("X", 1)]
            right_coeff = random.randint(2, 5)

            left_str = " + ".join(
                f"{c} {_fmt_term(parts)}" for c, parts in left_terms
            )
            right_str = f"{right_coeff} {_fmt_term(right_term)}"
            equation = f"{left_str} = {right_str}"

            var_lines = [f"{v} has dimension {_fmt_dim(dims[v])}" for v in sorted(dims)]
            var_desc = "; ".join(var_lines)

            answer = _fmt_vec(Xdim)

            metadata = edict({
                "vars": {v: _fmt_vec(dims[v]) for v in dims},
                "C": _fmt_vec(C),
                "Xdim": answer,
            })
            metadata.payload = {
                "var_desc": var_desc,
                "equation": equation,
            }
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        return (
            f"{p['var_desc']}.\n\n"
            f"In the equation\n\n    {p['equation']}\n\n"
            "all terms must have equal dimension for the equation to be dimensionally valid. "
            "X appears with power 1 and its dimension is the one missing piece. "
            "Express the dimension of X as a vector of exponents (m, l, t) where "
            "dim(X) = M^m L^l T^t. The answer is that triple of integers, e.g. (0,1,-1)."
        )

    def score_answer(self, answer, entry):
        parsed = parse_answer(answer, entry)
        if parsed is None:
            return 0.0
        truth = parse_answer(entry.answer, entry)
        return 1.0 if parsed == truth else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'dimensional_consistency (draw 2 of 2)',
 'hypothesis': 'W1-073',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/dimensional_consistency',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3968631907,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
