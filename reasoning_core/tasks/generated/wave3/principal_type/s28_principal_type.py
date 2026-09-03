import random
import string
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add principal-type inference over a tiny lambda calculus.',
 'hypothesis': 'S28',
 'changes': 'Ask for the principal type of a closed term, or whether it is '
            'typable.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3213085537,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class T:
    __slots__ = ('kind', 'a', 'b')
    def __init__(self, kind, a=None, b=None):
        self.kind = kind
        self.a = a
        self.b = b
    def __repr__(self):
        return _tfmt(self)


def TV(n):
    t = T('var')
    t.a = n
    return t


def FUN(a, b):
    return T('fun', a, b)


INT = T('int')
BOOL = T('bool')


_counter = [0]


def newvar():
    _counter[0] += 1
    return TV(_counter[0])


def apply_subst(subst, t):
    while t.kind == 'var' and t in subst:
        t = subst[t]
    if t.kind == 'fun':
        return FUN(apply_subst(subst, t.a), apply_subst(subst, t.b))
    return t


def unify(subst, a, b):
    a = apply_subst(subst, a)
    b = apply_subst(subst, b)
    if a.kind == 'var':
        if occurs(subst, a, b):
            return False
        subst[a] = b
        return True
    if b.kind == 'var':
        return unify(subst, b, a)
    if a.kind == 'fun' and b.kind == 'fun':
        return unify(subst, a.a, b.a) and unify(subst, a.b, b.b)
    if a.kind == 'int' and b.kind == 'int':
        return True
    if a.kind == 'bool' and b.kind == 'bool':
        return True
    return False


def occurs(subst, v, t):
    t = apply_subst(subst, t)
    if t.kind == 'var':
        return t is v
    if t.kind == 'fun':
        return occurs(subst, v, t.a) or occurs(subst, v, t.b)
    return False


def infer(expr, env, subst):
    k = expr[0]
    if k == 'var':
        if expr[1] in env:
            return apply_subst(subst, env[expr[1]])
        raise TypeError
    if k == 'lam':
        v = newvar()
        newenv = dict(env)
        newenv[expr[1]] = v
        body = infer(expr[2], newenv, subst)
        return FUN(v, body)
    if k == 'app':
        f = infer(expr[1], env, subst)
        a = infer(expr[2], env, subst)
        r = newvar()
        if not unify(subst, f, FUN(a, r)):
            raise TypeError
        return apply_subst(subst, r)
    if k == 'const':
        return INT if expr[1][0] == 'int' else BOOL
    if k == 'ite':
        c = infer(expr[1], env, subst)
        a = infer(expr[2], env, subst)
        b = infer(expr[3], env, subst)
        if not unify(subst, c, BOOL):
            raise TypeError
        if not unify(subst, a, b):
            raise TypeError
        return apply_subst(subst, a)
    raise TypeError


def principal(expr):
    _counter[0] = 0
    subst = {}
    try:
        t = infer(expr, {}, subst)
    except TypeError:
        return 'untypable'
    t = apply_subst(subst, t)
    return normalise(t)


def _tfmt(t):
    if t.kind == 'var':
        return 't%d' % t.a
    if t.kind == 'fun':
        inner = _tfmt(t.b)
        a = _tfmt(t.a)
        if t.a.kind == 'fun':
            a = '(' + a + ')'
        return a + ' -> ' + inner
    return t.kind


def norm_var_index(t, mapping, nexti):
    if t.kind == 'var':
        if t.a not in mapping:
            mapping[t.a] = nexti[0]
            nexti[0] += 1
        return ('v', mapping[t.a])
    if t.kind == 'fun':
        return ('fun', norm_var_index(t.a, mapping, nexti), norm_var_index(t.b, mapping, nexti))
    return (t.kind,)


def _fmt_norm(n, bound):
    if n[0] == 'v':
        return 't%d' % n[1]
    if n[0] == 'fun':
        inner = _fmt_norm(n[2], bound)
        a = _fmt_norm(n[1], bound)
        if n[1][0] == 'fun':
            a = '(' + a + ')'
        return a + ' -> ' + inner
    return n[0]


def normalise(t):
    root = norm_var_index(t, {}, [1])
    return _fmt_norm(root, {})


_CN = {'int': ['0', '1', '2', '3', '10'],
       'bool': ['true', 'false']}


def rand_const(rng):
    c = rng.choice(['int', 'int', 'int', 'bool'])
    return ('const', (c, rng.choice(_CN[c])))


def rand_expr(rng, depth, env):
    if depth <= 0:
        return rand_const(rng)
    r = rng.random()
    if r < 0.12 and env:
        return ('var', rng.choice(env))
    if r < 0.30:
        return rand_const(rng)
    if r < 0.62:
        v = rng.choice(string.ascii_lowercase)
        newenv = env + [v]
        return ('lam', v, rand_expr(rng, depth - 1, newenv))
    if r < 0.84:
        e1 = rand_expr(rng, depth - 1, env)
        e2 = rand_expr(rng, depth - 1, env)
        return ('app', e1, e2)
    c = rand_expr(rng, depth - 1, env)
    a = rand_expr(rng, depth - 1, env)
    b = rand_expr(rng, depth - 1, env)
    return ('ite', c, a, b)


def build_term(rng, depth):
    r = rng.random()
    if r < 0.08:
        return ('lam', 'x', ('var', 'x'))
    if r < 0.17:
        f = rng.choice('ab')
        g = rng.choice('cd')
        x = rng.choice('ef')
        return ('lam', f, ('lam', g,
               ('lam', x, ('app', ('app', ('var', f), ('var', x)), ('var', g)))))
    if r < 0.24:
        f = rng.choice('ab')
        x = rng.choice('cd')
        return ('lam', f, ('lam', x,
               ('app', ('var', f), ('app', ('var', f), ('var', x)))))
    if r < 0.32:
        f = rng.choice('ab')
        g = rng.choice('cd')
        x = rng.choice('ef')
        return ('lam', f, ('lam', g, ('lam', x,
               ('app', ('var', f), ('app', ('var', g), ('var', x))))))
    if r < 0.39:
        x = rng.choice('ab')
        y = rng.choice('cd')
        return ('lam', x, ('lam', y, ('ite', ('var', y),
               ('var', x), ('var', x))))
    if r < 0.46:
        f = rng.choice('ab')
        x = rng.choice('cd')
        g = rng.choice('ef')
        return ('lam', f, ('lam', g, ('lam', x,
               ('app', ('var', f), ('app', ('var', g), ('app', ('var', f),
                                                        ('var', x)))))))
    if r < 0.51:
        x = rng.choice('ab')
        return ('lam', x, ('ite', ('var', x), ('var', x),
               ('app', ('var', x), rand_const(rng))))
    if r < 0.56:
        f = rng.choice('ab')
        x = rng.choice('cd')
        return ('lam', f, ('lam', x,
               ('app', ('var', x), ('app', ('var', f), ('var', x)))))
    if r < 0.61:
        x = rng.choice('ab')
        y = rng.choice('cd')
        return ('lam', x, ('lam', y, ('var', x)))
    if r < 0.66:
        f = rng.choice('ab')
        g = rng.choice('cd')
        return ('lam', f, ('lam', g, ('app', ('var', g), ('var', f))))
    if r < 0.71:
        f = rng.choice('ab')
        g = rng.choice('cd')
        x = rng.choice('ef')
        return ('lam', f, ('lam', g, ('lam', x,
               ('app', ('var', g), ('app', ('var', f), ('app', ('var', g),
                                                        ('var', x)))))))
    if r < 0.76:
        x = rng.choice('ab')
        return ('lam', x, ('app', ('var', x), ('var', x)))
    return ('lam', 'x', rand_expr(rng, depth, ['x']))


def expr_to_prose(expr):
    k = expr[0]
    if k == 'var':
        return expr[1]
    if k == 'lam':
        return '\\' + expr[1] + '. ' + expr_to_prose(expr[2])
    if k == 'app':
        return '(' + expr_to_prose(expr[1]) + ' ' + expr_to_prose(expr[2]) + ')'
    if k == 'const':
        return expr[1][1]
    if k == 'ite':
        return '(if ' + expr_to_prose(expr[1]) + ' then ' + expr_to_prose(expr[2]) \
               + ' else ' + expr_to_prose(expr[3]) + ')'
    return '?'


@dataclass
class PrincipalTypeConfig(Config):
    max_depth: int = 5
    untypable_prob: float = 0.25

    def apply_difficulty(self, level):
        self.max_depth = sround(self.max_depth + level)


class PrincipalType(Task):
    config_cls = PrincipalTypeConfig

    def generate_entry(self):
        depth = int(self.config.max_depth)
        for _ in range(80):
            expr = build_term(random, depth)
            ans = principal(expr)
            if ans != 'untypable':
                return self._entry(expr, ans)
        expr = build_term(random, depth)
        ans = principal(expr)
        return self._entry(expr, ans)

    def _entry(self, expr, ans):
        untypable = ans == 'untypable'
        if not untypable and random.random() < self.config.untypable_prob:
            expr2 = ('app', expr, rand_const(random))
            ans2 = principal(expr2)
            if ans2 == 'untypable':
                expr, ans, untypable = expr2, ans2, True
        metadata = edict({
            'term': expr_to_prose(expr),
            'untypable': untypable,
            'max_depth': int(self.config.max_depth),
        })
        metadata.payload = {'term': metadata.term}
        if untypable:
            answer = 'untypable'
        else:
            answer = ans
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        base = (
            "Base types are 'int' and 'bool'. A type is written as "
            "alpha -> beta for a function from alpha to beta, right "
            "associative, and type variables are named t1, t2, ... in order "
            "of first appearance left to right."
        )
        return (
            f"{base}\n\n"
            f"Term: {metadata.term}\n\n"
            "The term is closed (no free variables), built from lambda "
            "abstraction \\x. e, application (f x), integer constants, boolean "
            "constants true/false, and if-then-else (if c then a else b) where "
            "c has type bool and a and b have the same type. Give the principal "
            "type of the term in the notation above, or answer exactly "
            "'untypable' if it has no type. The answer is the principal type, "
            "or 'untypable'."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        a = str(answer).strip()
        gold = entry.answer
        if gold == 'untypable':
            return 1.0 if a == 'untypable' else 0.0
        return 1.0 if a.replace(' ', '') == gold.replace(' ', '') else 0.0
