from dataclasses import dataclass
import random
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'Add exact polynomial gcd over the rationals.',
 'hypothesis': 'S54',
 'changes': 'Ask for the monic greatest common divisor of two polynomials.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4028055723,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _gcd_poly(a, b):
    a = [Fraction(c) for c in a]
    b = [Fraction(c) for c in b]

    def trim(p):
        while p and p[-1] == 0:
            p.pop()
        return p

    a = trim(a[:])
    b = trim(b[:])
    while b:
        aa = a[:]
        deg_a = len(aa) - 1
        deg_b = len(b) - 1
        lead_b = b[-1]
        while deg_a >= deg_b:
            coef = aa[deg_a] / lead_b
            shift = deg_a - deg_b
            for i in range(len(b)):
                aa[i + shift] -= coef * b[i]
            trim(aa)
            deg_a = len(aa) - 1
            if deg_a < deg_b:
                break
        a, b = b, aa
    if not a:
        a = [Fraction(1)]
    return a


def _monic(p):
    p = p[:]
    c = p[-1]
    if c != 1:
        p = [x / c for x in p]
    drop = 0
    while p and p[-1] == 0:
        p.pop()
        drop += 1
    return p, drop


def _rand_poly(rng, degree, coeff_range):
    while True:
        coeffs = [rng.randint(-coeff_range, coeff_range) for _ in range(degree + 1)]
        if coeffs[-1] != 0:
            return coeffs


def _render_poly(p):
    terms = []
    deg = len(p) - 1
    for i in range(len(p) - 1, -1, -1):
        c = p[i]
        if c == 0:
            continue
        sign = '+' if c > 0 else '-'
        abs_c = abs(c)
        if i == 0:
            term = str(abs_c)
        elif i == 1:
            if abs_c == 1:
                term = 'x'
            else:
                term = f'{abs_c}x'
        else:
            if abs_c == 1:
                term = f'x^{i}'
            else:
                term = f'{abs_c}x^{i}'
        terms.append((sign, term))
    if not terms:
        return '0'
    out = ''
    for idx, (sign, term) in enumerate(terms):
        prefix = '' if (idx == 0 and sign == '+') else sign
        out += f'{prefix}{term}'
    return out


def _render_fraction_coef(c):
    num = abs(c.numerator)
    den = c.denominator
    if den == 1:
        return str(num)
    return f'{num}/{den}'


def _render_poly_frac(p):
    terms = []
    for i in range(len(p) - 1, -1, -1):
        c = p[i]
        if c == 0:
            continue
        sign = '-' if c < 0 else '+'
        abs_c = abs(c)
        if i == 0:
            term = _render_fraction_coef(c)
        elif i == 1:
            if abs_c == 1:
                term = 'x'
            else:
                term = f'{_render_fraction_coef(abs_c)}x'
        else:
            if abs_c == 1:
                term = f'x^{i}'
            else:
                term = f'{_render_fraction_coef(abs_c)}x^{i}'
        terms.append((sign, term))
    if not terms:
        return '0'
    out = ''
    for idx, (sign, term) in enumerate(terms):
        prefix = '' if (idx == 0 and sign == '+') else sign
        out += f'{prefix}{term}'
    return out


@dataclass
class PolynomialGcdConfig(Config):
    degree: int = 3
    coeff_range: int = 6
    coprime_fraction: float = 0.2

    def apply_difficulty(self, level):
        self.degree = 2 + level
        self.coeff_range = 4 + 3 * level


class PolynomialGcd(Task):
    config_cls = PolynomialGcdConfig

    def generate_entry(self):
        cfg = self.config
        degree = cfg.degree
        coeff_range = cfg.coeff_range
        min_factor_deg = 1
        max_factor_deg = max(1, degree // 2)

        while True:
            factor_deg = random.randint(min_factor_deg, max_factor_deg)
            cofactor_deg = degree - factor_deg
            if cofactor_deg < 0:
                continue
            common = _rand_poly(random, factor_deg, coeff_range)
            a_co = _rand_poly(random, cofactor_deg, coeff_range)
            b_co = _rand_poly(random, cofactor_deg, coeff_range)

            def multiply(p, q):
                res = [0] * (len(p) + len(q) - 1)
                for i, cp in enumerate(p):
                    for j, cq in enumerate(q):
                        res[i + j] += cp * cq
                return res

            poly_a = multiply(common, a_co)
            poly_b = multiply(common, b_co)

            mon, _ = _monic([Fraction(c) for c in common])

            if random.random() < cfg.coprime_fraction:
                while True:
                    a = _rand_poly(random, degree, coeff_range)
                    b = _rand_poly(random, degree, coeff_range)
                    g = _gcd_poly([Fraction(c) for c in a],
                                  [Fraction(c) for c in b])
                    if len(g) == 1:
                        poly_a, poly_b = a, b
                        mon = [Fraction(1)]
                        break

            gcd_mon, _ = _monic(_gcd_poly([Fraction(c) for c in poly_a],
                                          [Fraction(c) for c in poly_b]))
            if gcd_mon != mon:
                continue

            p_a = _render_poly(poly_a)
            p_b = _render_poly(poly_b)
            answer = _render_poly_frac(mon)

            gcd_mon2 = _monic(_gcd_poly([Fraction(c) for c in poly_a],
                                        [Fraction(c) for c in poly_b]))[0]

            metadata = edict({
                'poly_a': p_a,
                'poly_b': p_b,
                'a_coeffs': [int(c) for c in poly_a],
                'b_coeffs': [int(c) for c in poly_b],
                'gcd': [str(Fraction(c)) for c in gcd_mon2],
            })
            metadata.payload = {'P': p_a, 'Q': p_b}
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"Find the monic greatest common divisor of the two polynomials below, "
            f"where each coefficient is written as a reduced fraction like 3/2 or -1/4 "
            f"(a whole number has no slash), and the leading term is written first with "
            f"its sign. Give only the resulting polynomial as the answer.\n\n"
            f"{render_payload(metadata.payload)}"
        )

    def score_answer(self, answer, entry):
        gt = entry.answer
        return 1.0 if answer == gt else 0.0
