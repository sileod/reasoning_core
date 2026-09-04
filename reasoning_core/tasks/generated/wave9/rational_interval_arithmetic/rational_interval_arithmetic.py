"""Exact rational interval arithmetic over closed intervals.

The task builds an expression tree whose leaves are closed intervals with
rational endpoints and whose internal nodes are one of {+, -, *, /,
intersection}. It evaluates the expression using exact rational interval
arithmetic and reports the tight resulting interval, or the empty set when an
intersection or a division by an interval containing zero empties the result.
"""

from dataclasses import dataclass

from fractions import Fraction
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


def _fmt(n):
    return str(n) if n.denominator == 1 else f"{n.numerator}/{n.denominator}"


def _fmt_interval(iv):
    if iv is None:
        return "empty"
    lo, hi = iv
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _fmt_node(node):
    kind, args = node
    if kind == "interval":
        return _fmt_interval(args)
    if kind == "div0":
        return f"1/{_fmt_node(args)}"
    name = {"add": "+", "sub": "-", "mul": "*", "div": "/", "int": "intersection"}[kind]
    return f"({name} {_fmt_node(args[0])} {_fmt_node(args[1])})"


def _safe_div(num, den):
    if den == 0:
        return None
    return num / den


def _iv_add(a, b):
    if a is None or b is None:
        return None
    return (a[0] + b[0], a[1] + b[1])


def _iv_sub(a, b):
    if a is None or b is None:
        return None
    return (a[0] - b[1], a[1] - b[0])


def _iv_mul(a, b):
    if a is None or b is None:
        return None
    vals = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (min(vals), max(vals))


def _iv_div(a, b):
    if a is None or b is None:
        return None
    if b[0] <= 0 <= b[1]:
        return None
    vals = [a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1]]
    return (min(vals), max(vals))


def _iv_int(a, b):
    if a is None or b is None:
        return None
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return (lo, hi)


def _eval_node(node):
    kind, args = node
    if kind == "interval":
        return args if args is not None else None
    if kind == "div0":
        a = _eval_node(args)
        if a is None or a[0] <= 0 <= a[1]:
            return None
        vals = [_safe_div(1, a[0]), _safe_div(1, a[1])]
        return (min(vals), max(vals))
    left = _eval_node(args[0])
    right = _eval_node(args[1])
    if kind == "add":
        return _iv_add(left, right)
    if kind == "sub":
        return _iv_sub(left, right)
    if kind == "mul":
        return _iv_mul(left, right)
    if kind == "div":
        return _iv_div(left, right)
    if kind == "int":
        return _iv_int(left, right)
    raise RuntimeError(kind)


def _rand_iv(max_radius):
    center = Fraction(random.randint(-6, 6), 1)
    radius = Fraction(random.randint(1, max_radius), 1)
    return (center - radius, center + radius)


def _rand_iv_pos(max_radius):
    lo = Fraction(random.randint(1, 5), 1)
    radius = Fraction(random.randint(1, max_radius), 1)
    return (lo, lo + radius)


EMPTY_OPS = ("add", "sub", "mul", "div")


def _gen_nonempty(leaf_radius, depth):
    leaf = ("interval", _rand_iv(leaf_radius))
    node = _build_nonempty(depth, leaf_radius, leaf)
    val = _eval_node(node)
    if val is None:
        raise RuntimeError("nonempty construction failed")
    return node, val


def _build_nonempty(depth, leaf_radius, leaf):
    if depth == 0:
        return leaf
    kind = random.choice(EMPTY_OPS)
    if kind == "div":
        return ("div", (leaf, ("interval", _rand_iv_pos(leaf_radius))))
    right = ("interval", _rand_iv(leaf_radius))
    return (kind, (_build_nonempty(depth - 1, leaf_radius, leaf), right))


def _gen_empty(leaf_radius):
    a = _rand_iv(leaf_radius)
    width = random.randint(1, leaf_radius)
    gap_lo = a[1] + Fraction(random.randint(1, leaf_radius), 1)
    b = (gap_lo, gap_lo + Fraction(width, 1))
    node = ("int", (("interval", a), ("interval", b)))
    val = _eval_node(node)
    if val is not None:
        raise RuntimeError("empty construction failed")
    return node, val


def _gen(leaf_radius, depth):
    for _ in range(100):
        if random.random() < 0.10:
            return _gen_empty(leaf_radius)
        return _gen_nonempty(leaf_radius, depth)
    return _gen_nonempty(leaf_radius, depth)


def _parse_interval(s):
    s = s.strip()
    if s == "empty":
        return None
    lo_s, hi_s = s[1:-1].split(",")
    def parse(t):
        t = t.strip()
        if "/" in t:
            num, den = t.split("/")
            return Fraction(int(num), int(den))
        return Fraction(int(t))
    return parse(lo_s), parse(hi_s)


@dataclass
class RationalIntervalConfig(Config):
    depth: int = 2
    max_radius: int = 3

    def apply_difficulty(self, level):
        self.depth = sround(self.depth + level)
        self.max_radius = sround(self.max_radius + 3 * level)


class RationalIntervalArithmetic(Task):
    summary = ("Propagate exact rational intervals through addition, subtraction, multiplication, "
               "division, and intersection; report the tight result or the empty set.")
    config_cls = RationalIntervalConfig

    def generate_entry(self):
        node, val = _gen(self.config.max_radius, self.config.depth)
        expr = _fmt_node(node)
        answer = _fmt_interval(val)
        metadata = edict({"expr": expr, "answer": answer})
        metadata.payload = {"expr": expr}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (render_payload(metadata.payload) + "\n\n"
                "Compute the exact result of this interval-arithmetic expression on closed intervals "
                "with rational endpoints. Operations: + (sum), - (difference), * (product), "
                "/ (quotient; undefined when the divisor contains zero), intersection (the overlap of "
                "the two intervals, empty when they do not overlap). Propagate values exactly. "
                "Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced "
                "fractions or integers, or the single word empty when the expression has no value. "
                "For example an intersection of [0, 1] and [2, 3] is empty.")

    def score_answer(self, answer, entry):
        try:
            got = _parse_interval(answer)
        except Exception:
            return 0.0
        want = _parse_interval(entry.answer)
        return 1.0 if got == want else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'rational_interval_arithmetic (draw 1 of 1)',
 'hypothesis': 'HV-064',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/rational_interval_arithmetic',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2382001662,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
