from reasoning_core.template import Problem, Task, edict, Config
from reasoning_core.utils import score_scalar
from unigram import init_grammar
from dataclasses import dataclass
import random
import unigram
import re
from decimal import Decimal, getcontext

getcontext().prec = 50

def _grammar():
    g = init_grammar(['py'], name="arithmetics", preprocess_template=lambda s:s)
    g('start(expr)',        '{0}')
    g('expr(expr)',       '({0})',            weight=1)
    g('expr(expr,expr)',  '{0} + {1}', weight=2)
    g('expr(expr,expr)',  '{0} - {1}', weight=1)
    g('expr(expr,expr)',  '{0} * {1}')
    g('expr(expr,expr)',  '{0} / {1}')
    g('expr(expr)',       '({0})**2',         weight=.25)
    g('expr(value)',       '{0}',weight= 10)
    g('value',  'NUM')
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
            return Problem(metadata=edict(expr=final_expr, height=x.height), answer=ans_str)
    
    def prompt(self, metadata):
        return f"Evaluate {metadata.expr}.\n Answer with only a number."

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)