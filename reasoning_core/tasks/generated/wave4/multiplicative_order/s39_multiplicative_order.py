import math
import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add multiplicative-order and discrete-logarithm reasoning in a prime '
         'modulus.',
 'hypothesis': 'S39',
 'changes': 'Ask for the order of an element modulo a prime, or the smallest '
            'exponent carrying one element to another.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 131861191,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def _factor(n):
    out = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def _divisors(prime_factors):
    out = [1]
    for p, e in prime_factors.items():
        base = out[:]
        mult = 1
        for _ in range(e):
            mult *= p
            out.extend(v * mult for v in base)
    return out


def _multiplicative_order(a, p):
    p1 = p - 1
    pf = _factor(p1)
    order = p1
    divs = _divisors(pf)
    for d in divs:
        if d < order and pow(a, d, p) == 1:
            order = d
    return order


def _factor_base_smooth_ok(p):
    pf = set(_factor(p - 1))
    return len(pf) >= 3


def _random_prime_in(low, high, rng):
    for _ in range(20000):
        n = rng.randrange(low, high + 1) | 1
        if n < 3 or not _is_prime(n):
            continue
        if _factor_base_smooth_ok(n):
            return n
        if len(_factor(n - 1)) >= 3:
            return n
    raise RuntimeError("could not find a suitable prime")


def _is_primitive_root(g, p):
    pf = set(_factor(p - 1))
    return all(pow(g, (p - 1) // q, p) != 1 for q in pf)


def _primitive_root(p, rng):
    for _ in range(20000):
        g = rng.randrange(2, p)
        if _is_primitive_root(g, p):
            return g
    raise RuntimeError("could not find a primitive root")


def _element_with_order(p, order, rng):
    _ = rng
    g = _primitive_root(p, random)
    return pow(g, (p - 1) // order, p)


def _bsgs(g, h, p, order):
    n = int(math.isqrt(order)) + 1
    table = {}
    e = 1
    for j in range(n):
        if e not in table:
            table[e] = j
        e = (e * g) % p
    factor = pow(g, (n * (p - 2)) % (p - 1), p) if False else pow(pow(g, n, p), p - 2, p)
    cur = h
    for i in range(n + 1):
        if cur in table:
            cand = i * n + table[cur]
            if pow(g, cand, p) == h:
                return cand
        cur = (cur * factor) % p
    return None


@dataclass
class MultiplicativeOrderConfig(Config):
    min_prime: int = 2000
    max_prime: int = 4000
    order_frac: float = 0.7
    suborder_p: float = 0.35

    def apply_difficulty(self, level):
        self.min_prime = int(1000 + 1200 * level ** 0.6)
        self.max_prime = int(3000 + 12000 * level ** 1.2)
        self.order_frac = min(0.95, 0.6 + 0.06 * level)
        self.suborder_p = 0.35


class MultiplicativeOrder(Task):
    summary = "Give the multiplicative order of an element modulo a prime, or the smallest exponent mapping one element to another."
    config_cls = MultiplicativeOrderConfig

    def generate_entry(self):
        minp = max(101, self.config.min_prime)
        maxp = self.config.max_prime
        order_frac = self.config.order_frac
        suborder_p = self.config.suborder_p

        mode = random.random() < 0.5

        if mode == 0:
            p = _random_prime_in(minp, maxp, random)
            pf = sorted(set(_factor(p - 1)))
            proper = random.random() < suborder_p and len(pf) >= 3
            if proper:
                d0 = random.choice(pf)
                reduced = (p - 1) // d0
                order = reduced
            else:
                order = p - 1
            a = _element_with_order(p, order, random)
            assert _multiplicative_order(a, p) == order
            answer = str(order)
            payload = {"question": "order", "prime": p, "element": a}
        else:
            p = _random_prime_in(minp, maxp, random)
            for _ in range(3000):
                g = random.randrange(2, p)
                if _is_primitive_root(g, p):
                    break
            assert _multiplicative_order(g, p) == p - 1
            h = random.randrange(1, p)
            k = _bsgs(g, h, p, p - 1)
            if k is None:
                answer = "none"
            else:
                answer = str(k)
            payload = {"question": "log", "prime": p, "base": g, "target": h}

        metadata = edict(payload)
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, m):
        if m.payload["question"] == "order":
            return (
                f"Let the modulus be the prime p = {m.payload['prime']}, and let a = {m.payload['element']}. "
                "The multiplicative order of a modulo p is the smallest positive integer n with a^n ≡ 1 (mod p). "
                "What is the multiplicative order of a modulo p? The answer is that integer."
            )
        else:
            return (
                f"Let the modulus be the prime p = {m.payload['prime']}, base g = {m.payload['base']}, "
                f"and target h = {m.payload['target']}. Find the smallest non-negative integer k such that "
                f"g^k ≡ h (mod p). If no such k exists, answer with the single word none. "
                "The answer is that integer, or the word none."
            )

    def score_answer(self, answer, entry):
        gold = entry.answer
        text = str(answer).strip().lower()
        if gold == "none":
            return float(text == "none")
        if text == "none":
            return 0.0
        try:
            return float(int(text) == int(gold))
        except (ValueError, TypeError):
            return 0.0
