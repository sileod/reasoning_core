import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'missing_dimension (draw 1 of 2)',
 'hypothesis': 'W1-074',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/missing_dimension',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2952937494,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

BASE_DIMS = ["L", "M", "T", "Q"]

# name -> base-dimension exponent vector (L, M, T, Q)
LIBRARY = [
    ("length l", (1, 0, 0, 0)),
    ("mass m", (0, 1, 0, 0)),
    ("time t", (0, 0, 1, 0)),
    ("electric charge q", (0, 0, 0, 1)),
    ("velocity v", (1, 0, -1, 0)),
    ("acceleration a", (1, 0, -2, 0)),
    ("force F", (1, 1, -2, 0)),
    ("energy E", (2, 1, -2, 0)),
    ("power P", (2, 1, -3, 0)),
    ("density rho", (-3, 1, 0, 0)),
    ("pressure p", (-1, 1, -2, 0)),
    ("momentum p0", (1, 1, -1, 0)),
    ("frequency f", (0, 0, -1, 0)),
    ("voltage V", (2, 1, -3, -1)),
    ("electric current I", (0, 0, -1, 1)),
    ("magnetic field B", (1, 0, -1, -1)),
    ("impedance Z", (2, 1, -3, -2)),
]


def _parse_vector(text):
    """Parse a comma-separated list of integers like '1,-1,2,0' into a list of ints."""
    try:
        cleaned = str(text).strip().replace(" ", "")
        if not cleaned:
            return None
        parts = cleaned.split(",")
        if not parts:
            return None
        vec = [int(p) for p in parts]
        if len(vec) != 4:
            return None
        return vec
    except Exception:
        return None


@dataclass
class MissingDimensionV1Config(Config):
    n_terms: int = 3
    exp_bound: int = 2

    def apply_difficulty(self, level):
        self.n_terms = sround(min(self.n_terms + level, 6))
        self.exp_bound = sround(min(self.exp_bound + level, 4))


class MissingDimension(Task):
    summary = "From a valid monomial formula with one unknown quantity dimension, output its base-dimension exponent vector over length, mass, time, charge, with terms and exponent signs varied."
    config_cls = MissingDimensionV1Config

    def generate_entry(self):
        n = min(self.config.n_terms, len(LIBRARY))
        expm = self.config.exp_bound
        while True:
            idxs = random.sample(range(len(LIBRARY)), n)
            names = [LIBRARY[i][0] for i in idxs]
            vecs = [LIBRARY[i][1] for i in idxs]
            coeffs = [random.choice([e for e in range(-expm, expm + 1) if e != 0]) for _ in range(n)]
            xv = [sum(coeffs[j] * vecs[j][d] for j in range(n)) for d in range(4)]
            if xv == [0, 0, 0, 0]:
                continue
            if max(max(xv), -min(xv)) > 8:
                continue
            if xv in vecs:
                continue
            break

        def power(name, coeff, neg_last):
            if coeff == 0:
                return ""
            if coeff == 1:
                return name
            if coeff == -1:
                return name + "^-1"
            r = f"{name}^{coeff}"
            if coeff < 0 and neg_last:
                r = f"{name}^({coeff})"
            return r

        terms = []
        negs = sum(1 for c in coeffs if c < 0)
        used_neg = False
        for j in range(n):
            r = power(names[j], coeffs[j], neg_last=False)
            if coeffs[j] < 0:
                used_neg = True
            terms.append(r)
        formula = " \\cdot ".join(t for t in terms)

        knowns = [{"name": names[j], "vector": list(vecs[j])} for j in range(n)]

        metadata = edict({
            "base_dims": BASE_DIMS,
            "knowns": knowns,
            "formula": formula,
            "unknown": "X",
            "vector": xv,
        })
        metadata.payload = {
            "Base dimensions": "[L, M, T, Q] = length, mass, time, electric charge (in that order).",
            "Known quantities": "\n".join(
                f"{k['name']} has base-dimension exponent vector "
                f"({k['vector'][0]}, {k['vector'][1]}, {k['vector'][2]}, {k['vector'][3]})."
                for k in knowns
            ),
            "Formula": f"A valid formula in these units is X = {formula}.",
            "Question": "The dimension of X is the only unknown quantity dimension. "
                        "Give the base-dimension exponent vector (eL, eM, eT, eQ) of X "
                        "as a comma-separated list of integers, e.g. 1,-1,2,0.",
        }
        answer = ",".join(str(e) for e in xv)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return render_payload(metadata.payload)

    def score_answer(self, answer, entry):
        gold = _parse_vector(entry.answer)
        pred = _parse_vector(answer)
        if pred is None or gold is None:
            return 0.0
        return 1.0 if pred == gold else 0.0
