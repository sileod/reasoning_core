from reasoning_core.template import Problem, Task, edict, Config
from reasoning_core.utils import score_scalar
from unigram import init_grammar
from dataclasses import dataclass
import random
import unigram
import re
from decimal import Decimal, getcontext
import ast, operator

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
    
    # Division is often excluded from basic symbolic simplification tasks to avoid fractions
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
    def update(self, c):
        self.min_depth += c
        self.max_depth += c
        self.out_digits += c
        self.out_decimals += c

def fill_num(expr, cfg=ArithmeticsConfig()):
    pat = re.compile(r'\bNUM\b')
    n = len(pat.findall(expr))
    def is_ok(v: Decimal):
        v = v.normalize()
        sign, digits, exponent = v.as_tuple()
        num_decimal_places = -exponent if exponent < 0 else 0
        if num_decimal_places > cfg.out_decimals:
            return False
        s_rep = f'{v:.{cfg.out_decimals}f}'
        total_digits = len(s_rep.replace('-', '').replace('.', ''))
        return total_digits <= cfg.out_digits

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
        if n > 1 and len(set(vals_str)) < 2:
            continue
        it = iter(f"Decimal('{x}')" for x in vals_str)
        e_decimal = pat.sub(lambda _: next(it), expr)
        try:
            v = eval(e_decimal, {"Decimal": Decimal})
        except Exception:
            continue
        if is_ok(v):
            it_str = iter(vals_str)
            final_expr_str = pat.sub(lambda _: next(it_str), expr)
            return final_expr_str, v
    raise RuntimeError('No assignment found; increase n_trials or widen pool.')

class Arithmetics(Task):
    def __init__(self, config=ArithmeticsConfig()):
        super().__init__(config=config)


    def generate(self):
        x = unigram.generate(g, depth=self.config.max_depth, min_depth=self.config.min_depth, mode=self.config.generation_algorithm)
        py_expr_template = x@'py'
        final_expr, value = fill_num(py_expr_template, cfg=self.config)
        quantizer = Decimal('1e-' + str(self.config.out_decimals))
        rounded_value = value.quantize(quantizer)
        ans_str = f"{rounded_value:f}".rstrip('0').rstrip('.')
        meta = edict(expr=final_expr, height=x.height)
        meta.cot = self.get_cot(final_expr)
        return Problem(metadata=meta, answer=ans_str)
    
    def prompt(self, metadata):
        return f"Evaluate {metadata.expr}.\n Answer with only a number."

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

    def get_cot(self, expr):
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
        syms = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}
        steps = []
    
        def visit(node):
            if isinstance(node, ast.Constant): 
                return Decimal(str(node.value))
            # Handle UnaryOp (negative numbers) silently
            if isinstance(node, ast.UnaryOp): 
                return -visit(node.operand)
            
            l, r = visit(node.left), visit(node.right)
            res = ops[type(node.op)](l, r)
            steps.append(f"{l} {syms[type(node.op)]} {r} = {res}")
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
        x = unigram.generate(g_sym, depth=self.config.max_depth, min_depth=self.config.min_depth)
        
        filler = lambda m: random.choice(self.config.variables) if m.group()=='VAR' else str(random.randint(1,9))
        final_expr = re.sub(r'\b(VAR|NUM)\b', filler, x @ 'py')
        
        # 2. Solve & Validate
        try:
            raw = parse_expr(final_expr, evaluate=False)
            simplified = sympy.simplify(raw)
            # Retry if trivial (no change or just a number)
            if raw == simplified or (simplified.is_number and not raw.is_number): return self.generate()
        except: return self.generate()

        meta = edict(expr=final_expr, cot=self.make_cot(raw))
        return Problem(metadata=meta, answer=str(simplified).replace('**', '^'))

    def make_cot(self, node):
        steps = []
        def visit(n):
            if n.is_Atom: return n
            # Bottom-up reconstruction
            new_n = n.func(*[visit(arg) for arg in n.args], evaluate=False)
            # Check for simplification opportunities
            simp = sympy.expand(new_n)
            if simp == new_n: simp = sympy.simplify(new_n)
            
            if simp != new_n: steps.append(f"{new_n} = {simp}")
            return simp
        
        visit(node)
        return "\n".join(steps)

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