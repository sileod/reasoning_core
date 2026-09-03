import random
import re

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'Add dimensional analysis over stated physical quantities.',
 'hypothesis': 'S47',
 'changes': 'Ask for the dimensions of a derived quantity, or which of several '
            'equations is dimensionally inconsistent.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1833805255,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


BASE_DIMS = {"M": [1, 0, 0, 0], "L": [0, 1, 0, 0], "T": [0, 0, 1, 0], "I": [0, 0, 0, 1]}
BASE_SYMBOLS = list("MLTI")

BASE_QUANTITIES = {
    "mass": "1,0,0,0",
    "a length": "0,1,0,0",
    "a time": "0,0,1,0",
    "current": "0,0,0,1",
    "velocity": "0,1,-1,0",
    "acceleration": "0,1,-2,0",
    "force": "1,1,-2,0",
    "momentum": "1,1,-1,0",
    "energy": "1,2,-2,0",
    "power": "1,2,-3,0",
    "frequency": "0,0,-1,0",
    "angular velocity": "0,0,-1,0",
    "pressure": "1,-1,-2,0",
    "density": "1,-3,0,0",
    "charge": "0,0,1,1",
    "voltage": "1,2,-3,-1",
    "resistance": "1,2,-3,-2",
    "capacitance": "-1,-2,4,2",
    "magnetic field": "1,0,-2,-1",
    "torque": "1,2,-2,0",
    "impulse": "1,1,-1,0",
    "wavelength": "0,1,0,0",
}

def _parse_fast(s):
    return [int(x) for x in s.split(",")]


def vec_to_str(v):
    parts = []
    for i, sym in enumerate(BASE_SYMBOLS):
        e = v[i]
        if e != 0:
            parts.append("%s^%d" % (sym, e))
    if not parts:
        return "M^0 L^0 T^0 I^0"
    return " ".join(parts)


_QUANT_STR = {k: vec_to_str(_parse_fast(v)) for k, v in BASE_QUANTITIES.items()}


def parse_vec(s):
    return _parse_fast(s)


def sc(vec, n):
    return [x * n for x in vec]


def add_vecs(a, b):
    return [x + y for x, y in zip(a, b)]


def expr_vec(tokens_factors, tokens_exps):
    v = [0, 0, 0, 0]
    for name, exp in zip(tokens_factors, tokens_exps):
        v = add_vecs(v, sc(parse_vec(BASE_QUANTITIES[name]), exp))
    return v


class DimConfig(Config):
    n_factors: int = 2
    exp_mag: int = 2
    n_choices: int = 4

    def apply_difficulty(self, level):
        import reasoning_core.template as tpl
        sround = tpl.stochastic_rounding
        self.n_factors = sround(self.n_factors + level)
        self.exp_mag = sround(self.exp_mag + level)
        self.n_choices = sround(4 + max(0, level - 1))


class DimensionalAnalysisTask(Task):
    task_name = "dimensional_analysis"
    config_cls = DimConfig

    def generate_entry(self):
        cfg = self.config
        names = list(BASE_QUANTITIES.keys())
        n_f = cfg.n_factors
        exp_mag = cfg.exp_mag

        while True:
            factors = [random.choice(names) for _ in range(n_f)]
            exps = [random.choice([-1, 1]) * random.randint(1, exp_mag) for _ in range(n_f)]
            v = expr_vec(factors, exps)
            if any(abs(e) > 0 for e in v):
                break

        expr_parts = []
        for name, e in zip(factors, exps):
            if e == 1:
                expr_parts.append(name)
            else:
                expr_parts.append("%s^%d" % (name, e))

        expr = " ".join(expr_parts)
        answer = vec_to_str(v)

        metadata = edict({
            "factors": factors,
            "exps": exps,
            "expr_str": expr,
            "answer_vec": [int(x) for x in v],
        })
        metadata.payload = {
            "quantities": {k: _QUANT_STR[k] for k in sorted(set(factors))},
            "expression": expr,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        base = (
            "The dimensions of a physical quantity are written as a product of powers of the "
            "base dimensions M (mass), L (length), T (time) and I (electric current). "
            "A product term with exponent zero is omitted; products are written as space-"
            "separated terms, each of the form SYMBOL^EXP, with exponents as integers.\n"
        )
        qtxt = "\n".join("  %s has dimensions %s" % (k, v) for k, v in metadata.payload["quantities"].items())
        payload = dict(metadata.payload)
        payload["quantities"] = qtxt
        expr = payload.pop("expression")
        return (
            base
            + "Given the following named quantities:\n"
            + qtxt
            + "\nWhat are the dimensions of the quantity " + expr
            + "?\nGive the answer in the canonical form described above, e.g. "
            + "M^1 L^2 T^-2, omitting terms with zero exponent.\n"
            + "The answer is the dimension vector string."
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        a = answer.replace(" ", "")
        g = gold.replace(" ", "")
        if a == g:
            return 1.0
        try:
            av = _parse_answer_vec(answer)
        except Exception:
            return 0.0
        if av == entry.metadata.answer_vec:
            return 1.0
        return 0.0


_VEC_RE = re.compile(r"([MLTI])\s*\^\s*(-?\d+)")


def _parse_answer_vec(s):
    out = {sym: 0 for sym in BASE_SYMBOLS}
    for m in _VEC_RE.finditer(s):
        out[m.group(1)] = int(m.group(2))
    return [out[c] for c in BASE_SYMBOLS]
