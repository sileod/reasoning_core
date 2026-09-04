"""Exact polynomial division and the Euclidean algorithm over finite fields or the rationals."""

import random
from dataclasses import dataclass

from sympy import Poly, Rational, symbols, gcd as sgcd

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

_x = symbols("x")


def _term_str(co, d):
    """Render a single monomial c*x**d as a canonical signed-free string."""
    co_s, neg = str(co), False
    if isinstance(co, (int, Rational)) and co < 0:
        neg = True
        co_s = str(-co)
    if d == 0:
        body = co_s
    elif d == 1:
        body = co_s + "*x" if co != 1 else "x"
    else:
        body = (co_s + "*x**" + str(d)) if co != 1 else ("x**" + str(d))
    return body, neg


def _poly_str(poly_or_expr, modulus=None):
    """Deterministically render a polynomial (monic gcd or prompt poly) as a string.

    Uses the poly's canonical (monom, coeff) arrays sorted by descending degree, so the
    output never depends on sympy's hash-seed-sensitive internal ordering.
    """
    poly = Poly(poly_or_expr, _x, modulus=modulus) if modulus else Poly(poly_or_expr, _x)
    terms = sorted(
        zip(poly.monoms(), poly.coeffs()),
        key=lambda mt: tuple(-d for d in mt[0]),
    )
    parts = []
    for mon, co in terms:
        d = mon[0]
        body, neg = _term_str(co, d)
        parts.append((body, neg))
    if not parts:
        return "0"
    out = []
    for i, (body, neg) in enumerate(parts):
        if i == 0:
            out.append(("-" if neg else "") + body)
        else:
            out.append((" - " if neg else " + ") + body)
    return "".join(out)


def _monic_str(poly_expr, modulus=None):
    """Return the canonical monic representative of ``poly_expr`` as a deterministic string."""
    expr = Poly(poly_expr, _x, modulus=modulus) if modulus else Poly(poly_expr, _x)
    if expr.degree() <= 0:
        return "1"
    lc = expr.LC()
    monic = expr * (Rational(1) / lc)
    return _poly_str(monic, modulus=modulus)


def _random_poly_int(max_deg, lo, hi):
    deg = random.randint(0, max_deg)
    coeffs = [random.randint(lo, hi) for _ in range(deg + 1)]
    return Poly.from_list(coeffs, _x)


def _random_poly_mod(max_deg, p):
    deg = random.randint(0, max_deg)
    coeffs = [random.randint(0, p - 1) for _ in range(deg + 1)]
    if coeffs[-1] == 0:
        coeffs[-1] = random.randint(1, p - 1)
    return Poly.from_list(coeffs, _x, modulus=p)


def _gcd_poly_mod(deg, p):
    coeffs = [random.randint(0, p - 1) for _ in range(deg + 1)]
    coeffs[-1] = 1
    return Poly.from_list(coeffs, _x, modulus=p)


def _gcd_poly_int(deg, lo, hi):
    coeffs = [random.randint(lo, hi) for _ in range(deg + 1)]
    coeffs[-1] = random.randint(1, hi)
    return Poly.from_list(coeffs, _x)


@dataclass
class PolyEuclidConfig(Config):
    base_max: int = 3
    extra_max: int = 3

    def apply_difficulty(self, level):
        self.base_max = sround(1 + level * 0.8)
        self.extra_max = sround(1 + level * 0.9)


class PolyEuclid(Task):
    summary = ("Execute exact polynomial division and the Euclidean algorithm over integers or "
               "finite fields, returning a canonical monic gcd.")
    config_cls = PolyEuclidConfig

    def generate_entry(self):
        p = random.choice(PRIMES)
        field = random.choice(["finite", "rational"])
        base_max = self.config.base_max
        extra_max = self.config.extra_max
        attempts = 0
        while True:
            attempts += 1
            if attempts > 200:
                raise RuntimeError("poly_euclid_algorithm: could not build a valid instance")
            dg = random.randint(1, base_max)
            if field == "finite":
                gcd_poly = _gcd_poly_mod(dg, p)
                a = _random_poly_mod(extra_max, p)
                b = _random_poly_mod(extra_max, p)
                f = (a * gcd_poly).as_expr()
                g = (b * gcd_poly).as_expr()
                actual = sgcd(f, g, modulus=p)
                gold = _monic_str(actual, modulus=p)
                fp = _poly_str(f, modulus=p)
                gp = _poly_str(g, modulus=p)
            else:
                gcd_poly = _gcd_poly_int(dg, 1, p)
                a = _random_poly_int(extra_max, 1, p)
                b = _random_poly_int(extra_max, 1, p)
                f = (a * gcd_poly).as_expr()
                g = (b * gcd_poly).as_expr()
                actual = sgcd(f, g, _x)
                gold = _monic_str(actual)
                fp = _poly_str(f)
                gp = _poly_str(g)
            if gold:
                break
        field_txt = f"the field F_{p}" if field == "finite" else "the rational numbers Q"
        metadata = edict({
            "field": field_txt,
            "p": int(p),
            "f": fp,
            "g": gp,
            "answer": gold,
        })
        metadata.payload = {
            "field": field_txt,
            "polynomial f": fp,
            "polynomial g": gp,
        }
        # Assert the defining property directly: gold must be monic and must divide both f and g
        # over the stated field.
        if field == "finite":
            ep = Poly(gold, _x, modulus=p)
            assert ep.LC() == 1, "gcd must be monic"
            assert Poly(f, _x, modulus=p).rem(ep).is_zero, "monic gcd must divide f"
            assert Poly(g, _x, modulus=p).rem(ep).is_zero, "monic gcd must divide g"
        else:
            eq = Poly(gold, _x)
            assert eq.LC() == 1, "gcd must be monic"
            assert Poly(f, _x).rem(eq).is_zero, "monic gcd must divide f"
            assert Poly(g, _x).rem(eq).is_zero, "monic gcd must divide g"
        return Entry(metadata=metadata, answer=gold)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            "Use the Euclidean algorithm over the stated field to find the great common divisor of "
            "these two polynomials, and return it in monic form (leading coefficient 1), written as "
            "a polynomial expression in the variable x. For example an answer looks like "
            "x**2 + 1, or x + 2, or 1. Give only the monic gcd expression as the answer."
        )

    def score_answer(self, answer, entry):
        expected = entry.answer
        if answer is None:
            return 0.0
        return 1.0 if str(answer).strip() == expected else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'polynomial_euclidean_algorithm (draw 1 of 1)',
 'hypothesis': 'HV-062',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/polynomial_euclidean_algorithm',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4112812501,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
