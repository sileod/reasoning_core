import math
import random
from dataclasses import dataclass

from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'Add continued-fraction reasoning over exact rationals.',
 'hypothesis': 'S26',
 'changes': 'Ask for a named convergent of a rational given by its '
            'continued-fraction expansion, or the expansion of a given '
            'rational.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3369804028,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ContinuedFractionsConfig(Config):
    max_coeff: int = 9
    min_coeff: int = 1
    max_len: int = 4

    def apply_difficulty(self, level):
        self.max_coeff = max(self.min_coeff, self.max_coeff + 4 * level)
        self.max_len = self.max_len + level


def _coefficients(fraction):
    coeffs = []
    while fraction != 0:
        n = math.floor(fraction)
        coeffs.append(n)
        frac_part = fraction - n
        if frac_part == 0:
            break
        fraction = 1 / frac_part
    return coeffs


def _convergent(coeffs, k):
    num, den = 1, 0
    p, q = 0, 1
    for i in range(k + 1):
        c = coeffs[i]
        num, p = c * num + p, num
        den, q = c * den + q, den
    return num, den


def _make_rational(config):
    length = random.randint(2, config.max_len)
    coeffs = [random.randint(config.min_coeff, config.max_coeff) for _ in range(length)]
    coeffs[0] = int(coeffs[0])
    value = Fraction(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        value = c + 1 / value
    return value, coeffs


@dataclass
class _Ctx:
    kind: str
    coeffs: list
    value: Fraction
    k: int
    num: int
    den: int


def _build(config):
    for _ in range(100):
        value, coeffs = _make_rational(config)
        if value <= 0 or value.denominator <= 0:
            continue
        kind = random.choice(["expansion", "convergent"])
        if kind == "expansion":
            return _Ctx(kind, coeffs, value, 0, 0, 0)
        k = random.randint(1, len(coeffs) - 1)
        num, den = _convergent(coeffs, k)
        if num > 0 and den > 0:
            return _Ctx(kind, coeffs, value, k, num, den)
    raise RuntimeError("could not build continued fraction instance")


class ContinuedFractions(Task):
    config_cls = ContinuedFractionsConfig

    def generate_entry(self):
        cfg = self.config
        ctx = _build(cfg)
        coeffs = ctx.coeffs
        if ctx.kind == "expansion":
            f = ctx.value
            metadata = edict({
                "payload": {
                    "prompt": (
                        "A rational number equals %d/%d. "
                        "What is its regular (simple) continued-fraction "
                        "expansion? Give the partial quotients as a "
                        "comma-separated list of integers."
                    ) % (f.numerator, f.denominator),
                },
                "kind": "expansion",
                "coeffs": coeffs,
            })
            answer = ",".join(str(c) for c in coeffs)
        else:
            f_num, f_den = ctx.num, ctx.den
            g = math.gcd(f_num, f_den)
            metadata = edict({
                "payload": {
                    "prompt": (
                        "A rational number has the continued-fraction "
                        "expansion [%s]. What is the convergent obtained "
                        "by truncating after the coefficient at index %d, "
                        "as a fraction in lowest terms (numerator then "
                        "denominator, separated by a slash)?"
                    ) % ("; ".join(str(c) for c in coeffs), ctx.k),
                },
                "kind": "convergent",
                "coeffs": coeffs,
                "k": ctx.k,
            })
            answer = "%d/%d" % (f_num // g, f_den // g)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return render_payload(metadata.payload)

    def score_answer(self, answer, entry):
        gold = entry.answer
        if answer is None:
            return 0.0
        got = str(answer).strip()
        if got == gold:
            return 1.0
        return 0.0
