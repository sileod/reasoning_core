from reasoning_core.template import Problem, Task, edict, Config
from reasoning_core.utils import score_scalar
from gramforge import init_grammar
from dataclasses import dataclass
import random
import gramforge
import re
from decimal import Decimal, getcontext, ROUND_HALF_UP
import ast, operator
from fractions import Fraction
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

getcontext().prec = 50

def _grammar(symbolic=False):
    g = init_grammar(['py'], name="arith", preprocess_template=lambda s:s)
    g('start(expr)',      '{0}')
    g('expr(expr)',       '({0})',              weight=1)
    g('expr(expr,expr)',  '{0} + {1}',          weight=2)
    g('expr(expr,expr)',  '{0} - {1}',          weight=1)
    g('expr(expr,expr)',  '{0} * {1}')
    g('expr(expr,expr)',  'max({0}, {1})',      weight=0.3)
    g('expr(expr,expr)',  'min({0}, {1})',      weight=0.3)
    g('expr(expr)',       'abs({0})',           weight=0.3)
    g('expr(expr)',       'round({0})',         weight=0.2)
    
    if not symbolic: 
        g('expr(expr,expr)', '{0} / {1}')
        
    g('expr(expr)',       '({0})**2',           weight=0.5 if symbolic else 0.25)
    g('expr(atom)',       '{0}',                weight=8 if symbolic else 10)
    
    g('atom', 'NUM')
    if symbolic: g('atom', 'VAR')
    return g

g=_grammar()

@dataclass
class ArithmeticsConfig(Config):
    min_depth: int = 3
    max_depth: int = 5
    generation_algorithm = "sequential"
    float_prob: float = 0.25
    in_decimals: int = 1
    out_decimals: int = 3
    out_digits: int = 6
    n_trials: int = 50_000
    trailing_zero_prob: float = 0.2
    trivial_prob = 0.1

    def update(self, c):
        self.min_depth += c
        self.max_depth += c
        self.out_digits += c
        self.out_decimals += c

def _add_trailing_zeros(s, prob=0.2):
    """Add trailing zeros to decimals with exponentially decreasing probability."""
    if '.' not in s: return s
    while random.random() < prob: s += '0'
    return s

def fill_num(expr, cfg=ArithmeticsConfig()):
    pat = re.compile(r'\bNUM\b')
    n = len(pat.findall(expr))
    def is_ok(v: Decimal):
        v = v.normalize()
        sign, digits, exponent = v.as_tuple()
        num_decimal_places = -exponent if exponent < 0 else 0
        if num_decimal_places > cfg.out_decimals: return False
        s_rep = f'{v:.{cfg.out_decimals}f}'
        return len(s_rep.replace('-', '').replace('.', '')) <= cfg.out_digits

    for _ in range(cfg.n_trials):
        vals_str = []
        has_division = '/' in expr
        for _ in range(n):
            if random.random() < cfg.float_prob:
                num = round(random.uniform(-12, 12), random.randint(1, cfg.in_decimals))
                if has_division and num == 0: num = random.choice([-1, 1])
                vals_str.append(str(num))
            else:
                num = random.randint(-15, 15)
                if has_division and num == 0: num = random.choice([-1, 1])
                vals_str.append(str(num))
        if n > 1 and len(set(vals_str)) < 2: continue
        
        it = iter(f"Decimal('{x}')" for x in vals_str)
        e_decimal = pat.sub(lambda _: next(it), expr)
        try:
            v = eval(e_decimal, {"Decimal": Decimal, "max": max, "min": min, "abs": abs, 
                                  "round": lambda x: x.to_integral_value(rounding=ROUND_HALF_UP)})
        except Exception: continue
        
        if is_ok(v):
            # Add trailing zeros for generalization
            vals_display = [_add_trailing_zeros(s, cfg.trailing_zero_prob) for s in vals_str]
            it_str = iter(vals_display)
            return pat.sub(lambda _: next(it_str), expr), v
    raise RuntimeError('No assignment found; increase n_trials or widen pool.')

class Arithmetics(Task):
    def __init__(self, config=ArithmeticsConfig()):
        super().__init__(config=config)

    def generate(self):
        x = gramforge.generate(g, depth=self.config.max_depth, min_depth=self.config.min_depth, mode=self.config.generation_algorithm)
        final_expr, value = fill_num(x@'py', cfg=self.config)
        quantizer = Decimal('1e-' + str(self.config.out_decimals))
        ans_str = f"{value.quantize(quantizer):f}".rstrip('0').rstrip('.')
        meta = edict(expr=final_expr, height=x.height, cot=self.get_cot(final_expr))
        return Problem(metadata=meta, answer=ans_str)
    
    def prompt(self, metadata):
        return f"Evaluate {metadata.expr}.\nAnswer with only a number."

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

    def get_cot(self, expr):
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, 
               ast.Div: operator.truediv, ast.Pow: operator.pow}
        syms = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}
        funcs = {'max': max, 'min': min, 'abs': abs, 'round': lambda x: Fraction(round(float(x)))}
        steps = []
        
        def fmt(n):
            d = n.denominator
            while d % 2 == 0: d //= 2
            while d % 5 == 0: d //= 5
            return f"{float(n):g}" if d == 1 else str(n)

        def visit(node):
            if isinstance(node, ast.Constant): return Fraction(str(node.value))
            if isinstance(node, ast.UnaryOp): return -visit(node.operand)
            if isinstance(node, ast.Call):
                fname = node.func.id
                args = [visit(a) for a in node.args]
                res = Fraction(funcs[fname](*args))
                steps.append(f"{fname}({', '.join(fmt(a) for a in args)}) = {fmt(res)}")
                return res
            l, r = visit(node.left), visit(node.right)
            res = ops[type(node.op)](l, r)
            steps.append(f"{fmt(l)} {syms[type(node.op)]} {fmt(r)} = {fmt(res)}")
            return res

        visit(ast.parse(expr, mode='eval').body)
        return "\n".join(steps)


@dataclass
class SymbolicConfig(ArithmeticsConfig):
    variables: tuple = ('x', 'y') # Start with just 2
    max_int: int = 9              # Start with single digits

def update(self, c):
        super().update(c) 
        
        self.max_int += int(10 * c)
        pool = "xyzabmnpqrstuvwdefghijkl"
        target_len = int(len(self.variables) + c)
        self.variables = tuple(pool[:min(len(pool), target_len)])

class SymbolicArithmetics(Task):
    def __init__(self, config=SymbolicConfig()):
        super().__init__(config=config)

    def generate(self):
        # 1. Generate & Fill
        g_sym = _grammar(symbolic=True)
        x = gramforge.generate(g_sym, depth=self.config.max_depth, min_depth=self.config.min_depth)
        
        filler = lambda m: random.choice(self.config.variables) if m.group()=='VAR' else str(random.randint(1,9))
        final_expr = re.sub(r'\b(VAR|NUM)\b', filler, x @ 'py')
        
        # 2. Solve & Validate
        trivial_allowed = random.random() < self.config.trivial_prob

        try:
            raw = parse_expr(final_expr, evaluate=False)
            simplified = sympy.simplify(raw)
            # Retry if trivial (no change or just a number)
            is_trivial = (raw == simplified) or (simplified.is_number and not raw.is_number)
            if is_trivial and not trivial_allowed: return self.generate()
        except: return self.generate()

        meta = edict(expr=final_expr, cot=self.make_cot(raw))
        return Problem(metadata=meta, answer=str(simplified).lower())


    def make_cot(self, node):
        steps = []
        def visit(n):
            if n.is_Atom: return n
            # Bottom-up reconstruction
            new_n = n.func(*[visit(arg) for arg in n.args], evaluate=False)
            # Check for simplification opportunities
            simp = sympy.expand(new_n)
            if simp == new_n: simp = sympy.simplify(new_n)
            s_old, s_new = str(new_n), str(simp)
            if s_old != s_new: steps.append(f"{new_n} = {simp}")
            return simp
        
        visit(node)
        return "\n".join(steps).lower()

    def prompt(self, meta):
            # Clean prompt: No CoT here
            return (f"Simplify the following algebraic expression:\n"
                    f"{meta.expr}\n\n"
                    f"Answer with the simplified expression.")
    def score_answer(self, answer, entry):
        try:
            clean = lambda s: str(s).split('=')[-1].strip().replace('^', '**')
            T = (standard_transformations + (implicit_multiplication_application,))
            diff = parse_expr(clean(answer), transformations=T) - parse_expr(clean(entry['answer']), transformations=T)
            return 1.0 if sympy.simplify(diff) == 0 else 0.0
        except: return 0.0