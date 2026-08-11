from reasoning_core.template import Entry, Task, DevTask, edict, Config, stochastic_rounding as sround
from reasoning_core.utils import score_scalar
from gramforge import init_grammar
from dataclasses import dataclass
import random
import gramforge
import re
import math
from decimal import Decimal, getcontext
import ast, operator
from fractions import Fraction
from itertools import combinations
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

getcontext().prec = 50

def _as_int(x):
    x = Fraction(x)
    if x.denominator != 1:
        raise ValueError("integer operation received a non-integer")
    return x.numerator

def _num_divisors(n):
    n = abs(_as_int(n))
    if n == 0:
        raise ValueError("divisor count undefined for 0")
    return int(sympy.divisor_count(n))


def _round_ties_to_even(x):
    """Round a rational exactly, using Python's nearest-even convention."""
    return Fraction(round(Fraction(x)))

SAFE_FUNCS = {
    "max": max,
    "min": min,
    "abs": abs,
    "round": _round_ties_to_even,
    "gcd": lambda a, b: math.gcd(_as_int(a), _as_int(b)),
    "lcm": lambda a, b: math.lcm(_as_int(a), _as_int(b)),
    "bit_count": lambda x: abs(_as_int(x)).bit_count(),
    "is_prime": lambda x: int(sympy.isprime(_as_int(x))),
    "prime_count": lambda x: int(sympy.primepi(_as_int(x))),
    "num_divisors": _num_divisors,
}
PYTHON_FUNCS = {**SAFE_FUNCS, "round": round}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
       ast.Pow: operator.pow, ast.Mod: operator.mod}
OP_NAMES = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "//", ast.Pow: "**", ast.Mod: "%"}

def _grammar(symbolic=False, division=True):
    g = init_grammar(['py'], name="arith", preprocess_template=lambda s:s)
    g('start(expr)',      '{0}')
    g('expr(expr)',       '({0})',              weight=1)
    g('expr(expr,expr)',  '{0} + {1}',          weight=2)
    g('expr(expr,expr)',  '{0} - {1}',          weight=1)
    g('expr(expr,expr)',  '{0} % {1}',          weight=1)
    g('expr(expr,expr)',  '{0} * {1}')
    g('expr(expr,expr)',  'max({0}, {1})',      weight=0.3)
    g('expr(expr,expr)',  'min({0}, {1})',      weight=0.3)
    g('expr(expr)',       'abs({0})',           weight=0.3)
    g('expr(expr)',       'round({0})',         weight=0.2)
    if not symbolic:
        g('expr(int,int)',   'gcd({0}, {1})',   weight=0.25)
        g('expr(pos,pos)',   'lcm({0}, {1})',   weight=0.2)
        g('expr(nat)',       'bit_count({0})',  weight=0.2)
        g('expr(nat)',       'is_prime({0})',   weight=0.15)
        g('expr(nat)',       'prime_count({0})', weight=0.15)
        g('expr(pos)',       'num_divisors({0})', weight=0.15)
    
    if division and not symbolic:
        g('expr(expr,expr)', '{0} / {1}')
    if division:
        g('expr(expr,expr)', '{0} // {1}')
        
    g('expr(expr)',       '({0})**2',           weight=0.5 if symbolic else 0.25)
    g('expr(atom)',       '{0}',                weight=8 if symbolic else 10)
    
    g('atom', 'NUM')
    g('int', 'INT')
    g('nat', 'NAT')
    g('pos', 'POS')
    if symbolic: g('atom', 'VAR')
    return g

g=_grammar()

@dataclass
class ArithmeticsConfig(Config):
    min_depth: int = 3
    max_depth: int = 5
    gramforge_algorithm = "sequential"
    float_prob: float = 0.25
    in_decimals: int = 1
    out_decimals: int = 3
    out_digits: int = 6
    n_trials: int = 50_000
    trailing_zero_prob: float = 0.2
    trivial_prob = 0.01
    bool_prob = 0.1
    spaced_digits_prob: float = 0.03
    reversed_spaced_digits_prob: float = 0.02
    semantics: str = "exact"
    semantic_decoy_prob: float = 0.15

    def apply_difficulty(self, level):
        self.min_depth = sround(self.min_depth + level)
        self.max_depth = sround(self.max_depth + level)
        self.out_digits = sround(self.out_digits + level)
        self.out_decimals = sround(self.out_decimals + level)

def _add_trailing_zeros(s, prob=0.2):
    """Add trailing zeros to decimals with exponentially decreasing probability."""
    if '.' not in s: return s
    while random.random() < prob: s += '0'
    return s


def _evaluate(expr, semantics="exact", steps=None, ties=None):
    """Evaluate the task grammar with exact rationals or native Python numbers."""
    if semantics not in {"exact", "python"}:
        raise ValueError(f"Unknown arithmetic semantics: {semantics}")
    funcs = SAFE_FUNCS if semantics == "exact" else PYTHON_FUNCS

    def fmt(x):
        if semantics == "python": return repr(x) if isinstance(x, float) else str(x)
        x = Fraction(x)
        d = x.denominator
        while d % 2 == 0: d //= 2
        while d % 5 == 0: d //= 5
        return f"{float(x):g}" if d == 1 else str(x)

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value)) if semantics == "exact" else node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [visit(a) for a in node.args]
            if node.func.id == "round" and abs(args[0]) % 1 == Fraction(1, 2):
                if ties is not None: ties.append(True)
            out = funcs[node.func.id](*args)
            out = Fraction(out) if semantics == "exact" else out
            if steps is not None: steps.append(f"{node.func.id}({', '.join(map(fmt, args))}) = {fmt(out)}")
            return out
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a, b = visit(node.left), visit(node.right)
            out = OPS[type(node.op)](a, b)
            out = Fraction(out) if semantics == "exact" else out
            if steps is not None: steps.append(f"{fmt(a)} {OP_NAMES[type(node.op)]} {fmt(b)} = {fmt(out)}")
            return out
        raise ValueError("Unsupported arithmetic expression")

    return visit(ast.parse(expr, mode="eval").body)


def _output_decimal(value, cfg):
    if isinstance(value, Fraction):
        d = value.denominator
        while d % 2 == 0: d //= 2
        while d % 5 == 0: d //= 5
        if d != 1: return None
        dec = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float) and math.isfinite(value):
        dec = Decimal(repr(value))
    else:
        return None
    dec = dec.normalize()
    if max(0, -dec.as_tuple().exponent) > cfg.out_decimals: return None
    shown = f"{dec:.{cfg.out_decimals}f}"
    return dec if len(shown.replace("-", "").replace(".", "")) <= cfg.out_digits else None


def fill_num(expr, cfg=None):
    cfg = cfg or ArithmeticsConfig()
    pat = re.compile(r'\b(NUM|INT|NAT|POS)\b')
    tokens = pat.findall(expr)

    has_division = '/' in expr
    for _ in range(cfg.n_trials):
        vals_str = []
        for tok in tokens:
            r = random.random()
            if tok == "INT":                           num = random.randint(-30, 30)
            elif tok == "NAT":                         num = random.randint(0, 80)
            elif tok == "POS":                         num = random.randint(1, 80)
            elif r < cfg.bool_prob:                    num = random.randint(0, 1)
            elif r < cfg.bool_prob + cfg.float_prob:   num = round(random.uniform(-12, 12), random.randint(1, cfg.in_decimals))
            else:                                      num = random.randint(-15, 15)
            if tok == "NUM" and has_division and num == 0: num = random.choice([-1, 1])
            vals_str.append(str(num))

        values = iter(vals_str)
        final_expr = pat.sub(lambda _: next(values), expr)
        try:
            value = _evaluate(final_expr, cfg.semantics)
        except (ArithmeticError, OverflowError, TypeError, ValueError):
            continue

        dec = _output_decimal(value, cfg)
        if dec is not None:
            it_str = iter(_add_trailing_zeros(s, cfg.trailing_zero_prob) for s in vals_str)
            return pat.sub(lambda _: next(it_str), expr), dec
    raise RuntimeError('No assignment found; increase n_trials or widen pool.')

def _space_number(m, reverse=False):
    s = m.group()
    chars = list(reversed(s)) if reverse else list(s)
    return " ".join(chars)


def _display_expr(expr, answer, cfg):
    tree = ast.parse(expr, mode="eval")
    awkward = "." in answer or answer.startswith("-") or any(
        isinstance(n, ast.Constant) and isinstance(n.value, float)
        or isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
        for n in ast.walk(tree)
    )
    if awkward or not any(len(n) > 1 for n in re.findall(r"\d+", expr) + [answer]):
        return expr, "normal"
    r = random.random()
    if r < cfg.spaced_digits_prob:
        return re.sub(r"\d+(?:\.\d+)?", _space_number, expr), "spaced"
    if r < cfg.spaced_digits_prob + cfg.reversed_spaced_digits_prob:
        return re.sub(r"\d+(?:\.\d+)?", lambda m: _space_number(m, True), expr), "reversed_spaced"
    return expr, "normal"


def _format_number(s, mode):
    if mode == "spaced":
        return re.sub(r"\d+(?:\.\d+)?", _space_number, s)
    if mode == "reversed_spaced":
        return re.sub(r"\d+(?:\.\d+)?", lambda m: _space_number(m, True), s)
    return s


def _canonical_decimal(value, decimal_places):
    quantizer = Decimal(f"1e-{decimal_places}")
    quantized = value.quantize(quantizer)
    if quantized.is_zero():
        return "0"
    return f"{quantized:f}".rstrip("0").rstrip(".")


def _canonical_value(value, places):
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, float):
        value = Decimal(repr(value))
    else:
        value = Decimal(value)
    return _canonical_decimal(value, places)


def _semantic_cue_required(expr, places):
    try:
        return _canonical_value(_evaluate(expr, "exact"), places) != \
               _canonical_value(_evaluate(expr, "python"), places)
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        return True


def _semantic_decoy_eligible(expr):
    tree = ast.parse(expr, mode="eval")
    decimal = any(isinstance(n, ast.Constant) and isinstance(n.value, float) for n in ast.walk(tree))
    relevant = any(
        isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod))
        or isinstance(n, ast.Call) and getattr(n.func, "id", None) == "round"
        for n in ast.walk(tree)
    )
    return decimal and relevant


def _contextual_semantics(expr, semantics="exact", places=3, decoy=False):
    cues = []
    if _semantic_cue_required(expr, places) or decoy and _semantic_decoy_eligible(expr):
        cues.append("Use exact arithmetic." if semantics == "exact"
                    else "Use Python floating-point semantics.")
    ties = []
    _evaluate(expr, semantics, ties=ties)
    if ties: cues.append("Round ties to the nearest even integer.")
    return cues


class Arithmetics(Task):
    config_cls = ArithmeticsConfig
    summary = "Compositional arithmetics with float/int/bool, varied operators, number theory."

    def generate_entry(self):
        while True:
            x = gramforge.generate(g, depth=self.config.max_depth, min_depth=self.config.min_depth, mode=self.config.gramforge_algorithm)
            expr = x@'py'
            if expr.count('NUM') > 1 or random.random() < self.config.trivial_prob: break
        final_expr, value = fill_num(expr, cfg=self.config)
        ans_str = _canonical_decimal(value, self.config.out_decimals)
        shown_expr, digit_mode = _display_expr(final_expr, ans_str, self.config)
        cue = _semantic_cue_required(final_expr, self.config.out_decimals)
        if not cue and _semantic_decoy_eligible(final_expr):
            cue = random.random() < self.config.semantic_decoy_prob
        meta = edict(expr=final_expr, display_expr=shown_expr, digit_mode=digit_mode,
                     semantics=self.config.semantics, semantic_cue=cue,
                     out_decimals=self.config.out_decimals, height=x.height,
                     cot=self.get_cot(final_expr))
        return Entry(metadata=meta, answer=_format_number(ans_str, digit_mode))
    
    def render_prompt(self, metadata):
        note = {
            "spaced": " Digits are spaced; answer likewise.",
            "reversed_spaced": " Digits are reversed and spaced; answer likewise.",
        }.get(metadata.get("digit_mode"), "")
        cues = _contextual_semantics(
            metadata.expr,
            metadata.get("semantics", self.config.semantics),
            metadata.get("out_decimals", self.config.out_decimals),
            metadata.get("semantic_cue", False),
        )
        semantics = "".join(f"\n{cue}" for cue in cues)
        return f"Evaluate {metadata.get('display_expr', metadata.expr)}.{note}{semantics}\nThe answer is a number."

    def score_answer(self, answer, entry):
        if entry.metadata.get("digit_mode", "normal") == "normal":
            return score_scalar(answer, entry)
        return float(str(answer).strip() == str(entry.answer).strip())

    def get_cot(self, expr):
        steps = []
        _evaluate(expr, self.config.semantics, steps=steps)
        return "\n".join(steps)



import random, sympy as sp
from dataclasses import dataclass
from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.utils import score_scalar


UNITS = "stamps marbles coins books apples cards tokens beads tiles cookies shells stickers pebbles buttons".split()
NAMES = ("Mara Jon Aisha Wei Sofia Diego Priya Tom Lena Omar Yuki Hana Carlos Nina "
         "Zara Ravi Mei Iris Noah Amara Leo Sana Kof Tara").split()

ORD = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth"}
MUL = {2: "doubled", 3: "tripled", 4: "quadrupled"}


@dataclass
class WordProblemMathConfig(Config):
    n_rel: int = 2
    max_n: int = 12
    inverse_p: float = .30
    relational_p: float = .50
    deep_query_p: float = .15

    def apply_difficulty(self, level):
        self.n_rel = sround(self.n_rel + level)
        self.max_n = sround(self.max_n + 12 * level)
        self.inverse_p = min(.70, self.inverse_p + .08 * level)
        self.deep_query_p = min(.85, self.deep_query_p + .15 * level)


def ri(a, b):
    return random.randint(int(a), int(b))


def unique(expr, observed, x, expect):
    sols = sp.solve(sp.Eq(expr, observed), x, dict=True)
    if len(sols) != 1 or x not in sols[0]:
        return False
    xv = sp.nsimplify(sols[0][x])
    return bool(xv.is_integer) and int(xv) == expect


def clean_answer(a):
    return isinstance(a, int) and 0 < a < 6000


def process_step_text(step, unit):
    op, k = step
    if op == "mul":
        return MUL[k] if random.random() < .6 else f"multiplied by {k}"
    if op == "div":
        return f"cut to {ORD[k]}"
    if op == "add":
        return f"{k} more {unit} added"
    return f"{k} {unit} removed"


def relation_text(rel, unit):
    op, a, b, k, c = rel
    if op == "times":
        return f"{a} has {k} times as many {unit} as {b}"
    if op == "more":
        return f"{a} has {k} more {unit} than {b}"
    if op == "fewer":
        return f"{a} has {k} fewer {unit} than {b}"
    if op == "frac":
        return f"{a} has {ORD[k]} as many {unit} as {b}"
    return f"{a} has as many {unit} as {b} and {c} combined"


def relational_proof_core_size(names, rels, given, asked, given_value):
    """Minimum relation equations that identify ``asked`` from ``given``."""
    symbols = {name: sp.Symbol(name) for name in names}
    equations = []
    for op, a, b, k, c in rels:
        rhs = symbols[b] + symbols[c] if op == "combine" else {
            "times": k * symbols[b],
            "more": symbols[b] + k,
            "fewer": symbols[b] - k,
            "frac": symbols[b] / k,
        }[op]
        equations.append(sp.Eq(symbols[a], rhs))

    variables = [symbols[name] for name in names]
    asked_index = names.index(asked)
    given_eq = sp.Eq(symbols[given], given_value)
    for size in range(len(equations) + 1):
        for subset in combinations(equations, size):
            matrix, _ = sp.linear_eq_to_matrix((given_eq, *subset), variables)
            if all(vector[asked_index] == 0 for vector in matrix.nullspace()):
                return size
    return None


def gen_process(config):
    unit = random.choice(UNITS)
    x = sp.Symbol("x", positive=True)
    base = ri(2, config.max_n)
    expr, cur, steps = x, sp.Integer(base), []

    for _ in range(ri(2, 2 + config.n_rel)):
        op = random.choice(["add", "sub", "mul", "div"])

        if op == "mul":
            k = ri(2, 4)
            expr, cur = k * expr, k * cur

        elif op == "div":
            ks = [k for k in (2, 3, 4, 5) if int(cur) % k == 0 and int(cur) // k >= 2]
            if not ks:
                continue
            k = random.choice(ks)
            expr, cur = expr / k, cur // k

        elif op == "add":
            k = ri(2, config.max_n)
            expr, cur = expr + k, cur + k

        else:
            if int(cur) <= 3:
                continue
            k = ri(2, int(cur) - 1)
            expr, cur = expr - k, cur - k

        steps.append((op, k))

    if len(steps) < 2 or int(cur) < 2:
        return None

    observed = int(cur)
    inverse = random.random() < config.inverse_p
    answer = base if inverse else observed

    if inverse and not unique(expr, observed, x, base):
        return None

    metadata = edict(
        family="process",
        unit=unit,
        base=base,
        observed=observed,
        inverse=inverse,
        steps=steps,
        expr=str(expr),
        equation=str(sp.Eq(expr, observed))
    )
    return Entry(metadata=metadata, answer=str(answer))


def gen_relational(config):
    unit = random.choice(UNITS)
    m = ri(3, min(6, 3 + config.n_rel))
    names = random.sample(NAMES, m)

    x = sp.Symbol("x", positive=True)
    val, rels = {names[0]: x}, []

    for i in range(1, m):
        a = names[i]
        parents = names[:i]
        ops = ["times", "more", "fewer", "frac"] + (["combine"] if len(parents) >= 2 else [])
        op = random.choice(ops)

        if op == "times":
            b, k = random.choice(parents), ri(2, 4)
            val[a] = k * val[b]
            rels.append((op, a, b, k, None))

        elif op == "more":
            b, k = random.choice(parents), ri(2, config.max_n)
            val[a] = val[b] + k
            rels.append((op, a, b, k, None))

        elif op == "fewer":
            b, k = random.choice(parents), ri(2, config.max_n)
            val[a] = val[b] - k
            rels.append((op, a, b, k, None))

        elif op == "frac":
            b, k = random.choice(parents), random.choice([2, 3, 4])
            val[a] = val[b] / k
            rels.append((op, a, b, k, None))

        else:
            b, c = random.sample(parents, 2)
            val[a] = val[b] + val[c]
            rels.append((op, a, b, None, c))

    base = nums = None
    candidates = list(range(2, int(config.max_n) + 1))
    random.shuffle(candidates)

    for cand in candidates[:24]:
        cur = {k: sp.nsimplify(v.subs(x, cand)) for k, v in val.items()}
        if all(t.is_integer and t > 0 for t in cur.values()):
            base = cand
            nums = {k: int(v) for k, v in cur.items()}
            break

    if base is None:
        return None

    revealable = [z for z in names if nums[z] >= 2]
    if not revealable:
        return None

    pairs = []
    for given in revealable:
        if not unique(val[given], nums[given], x, base):
            continue
        for asked in names:
            if asked == given:
                continue
            core_size = relational_proof_core_size(
                names, rels, given, asked, nums[given]
            )
            if core_size is not None:
                pairs.append((given, asked, core_size))
    require_deep = random.random() < config.deep_query_p
    eligible = [pair for pair in pairs if pair[2] >= 2] if require_deep else pairs
    if not eligible:
        return None
    given, asked, core_size = random.choice(eligible)

    random.shuffle(rels)

    metadata = edict(
        family="relational",
        unit=unit,
        names=names,
        relations=rels,
        given=given,
        asked=asked,
        given_value=nums[given],
        values=nums,
        base=base,
        query_distance=core_size,
        proof_core_size=core_size,
        deep_query_required=require_deep,
        equation=str(sp.Eq(val[given], nums[given]))
    )
    return Entry(metadata=metadata, answer=str(nums[asked]))


class MathWordProblem(Task):
    summary = "Solve relational and process math word problems involving objects and values."
    def __init__(self, config=None):
        super().__init__(config=config or WordProblemMathConfig())

    def generate_entry(self):
        for _ in range(100):
            gen = gen_relational if random.random() < self.config.relational_p else gen_process
            problem = gen(self.config)
            if problem and clean_answer(int(problem.answer)):
                return problem
        return None

    def render_prompt(self, m):
        if m.family == "process":
            chain = "; then ".join(process_step_text(s, m.unit) for s in m.steps)
            if m.inverse:
                return (
                    f"A jar holds some {m.unit}. {chain}. "
                    f"The jar now holds {m.observed} {m.unit}. "
                    f"How many {m.unit} did it start with? Answer with a number."
                )
            return (
                f"A jar holds {m.base} {m.unit}. {chain}. "
                f"How many {m.unit} are in the jar now? Answer with a number."
            )

        lines = ". ".join(relation_text(r, m.unit) for r in m.relations)
        return (
            f"{lines}. {m.given} has {m.given_value} {m.unit}. "
            f"How many {m.unit} does {m.asked} have? Answer with a number."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

    def balancing_key(self, problem):
        m = problem.metadata
        if m.family == "process":
            ops = ",".join(op for op, _ in m.steps)
            return f"process:{'inverse' if m.inverse else 'forward'}:{ops}"
        ops = ",".join(r[0] for r in m.relations)
        return f"relational:d{min(m.proof_core_size, 3)}:{ops}"

    def deduplication_key(self, problem):
        m = problem.metadata
        if m.family == "process":
            return str((m.family, m.unit, m.base, m.observed, tuple(map(tuple, m.steps)), m.inverse))
        return str((m.family, m.unit, tuple(map(tuple, m.relations)), m.given, m.given_value, m.asked))
