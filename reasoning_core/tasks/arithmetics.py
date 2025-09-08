from reasoning_core.template import Problem, Task, edict, ProceduralDataset, Config
from unigram import init_grammar
import unigram
import num2words
from dataclasses import dataclass
import random
import re
import exrex
from decimal import Decimal, getcontext
from reasoning_core.utils import score_scalar


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
class ArithConfig(Config):
    min_depth: int = 5
    max_depth: int = 8
    generation_algorithm = "sequential"
    float_prob: float = 0.25

    def update(self, c):
        self.min_depth += c
        self.max_depth += c
        self.out_digits += c
        self.out_decimals += c

    in_decimals: int = 1
    out_decimals: int = 3
    out_digits: int = 6
    n_trials: int = 50_000


def fill_num(expr, cfg=ArithConfig()):
    has_division = '/' in expr
    pat = re.compile(r'\bNUM\b')
    n = len(pat.findall(expr))
    fmt = str
    ok  = lambda v: (any(abs(v*10**k - round(v*10**k)) < 1e-9 for k in range(cfg.out_decimals+1)) and
                     len(f'{v:.{cfg.out_decimals}f}'.replace('-','').replace('.','')) <= cfg.out_digits)

    for _ in range(cfg.n_trials):
        vals = [
            (lambda v: v if not (has_division and v == 0) else random.choice([-1, 1]))(
                round(random.uniform(-12, 12), random.randint(1, cfg.in_decimals))
                if random.random() < cfg.float_prob
                else random.randint(-15, 15)
            )
            for _ in range(n)
        ]
        if n > 1 and len({round(x, cfg.in_decimals) for x in vals}) < 2:
            continue
        it = iter(fmt(x) for x in vals)
        e = pat.sub(lambda _: next(it), expr)
        try:
            v = eval(e, {'__builtins__': None}, {})
        except Exception:
            continue
        if ok(v):
            return e
    raise RuntimeError('No assignment found; increase n_trials or widen pool.')

def exact_eval(expression_string: str) -> Decimal:
    getcontext().prec = 100
    decimal_expr = re.sub(r'(\d+\.?\d*)', r"Decimal('\1')", expression_string)
    return eval(decimal_expr, {"Decimal": Decimal}).normalize()


class Arith(Task):
    def __init__(self, config=ArithConfig()):
        super().__init__(config=config)

    def generate(self):
        x=unigram.generate(g, depth=self.config.max_depth, min_depth=self.config.min_depth, mode=self.config.generation_algorithm)
        expr = x@'py'
        expr = fill_num(expr, cfg=self.config)
        value = exact_eval(expr)
        return Problem(metadata=edict(expr=expr, height=x.height), answer=str(value))

    def prompt(self, metadata):
        p = (
            f"Evaluate {metadata.expr}.\n Answer with only a number."
        )
        return p

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)