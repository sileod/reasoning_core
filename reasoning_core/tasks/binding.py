"""Symbolic rewriting: λ-calculus β-reduction and first-order unification.

Two tasks over symbolic expressions with binding / meta-variables:

- `lambda_reduction` — reduce an untyped λ-term to β-normal form.
  Named-variable representation with capture-avoiding substitution; answers are
  compared up to α-equivalence by canonicalising to de Bruijn indices.

- `term_unification` — find the most general unifier of two first-order terms,
  backed by the `unification` library (logpy / kanren). Generation builds a
  shared skeleton and two partial instantiations, guaranteeing unifiability;
  instances whose MGU retains free variables are rejected to keep scoring
  unambiguous.
"""
from dataclasses import dataclass
import ast, random, re

from unification import unify, reify, var as mkvar
from unification.variable import Var

from reasoning_core.template import Task, Problem, Config, edict


# ─────────────────────────────────────────────────────────────────────────────
# λ-calculus core
# ─────────────────────────────────────────────────────────────────────────────
# term ::= ('v', name) | ('l', name, body) | ('a', fun, arg)

def _fv(t):
    k = t[0]
    if k == 'v': return {t[1]}
    if k == 'l': return _fv(t[2]) - {t[1]}
    return _fv(t[1]) | _fv(t[2])

def _fresh(avoid):
    i = 0
    while (n := f"_{i}") in avoid: i += 1
    return n

def _subst(t, x, s):
    k = t[0]
    if k == 'v': return s if t[1] == x else t
    if k == 'a': return ('a', _subst(t[1], x, s), _subst(t[2], x, s))
    y, body = t[1], t[2]                         # k == 'l'
    if y == x: return t
    if y in _fv(s):
        y2 = _fresh(_fv(body) | _fv(s) | {x})
        body, y = _subst(body, y, ('v', y2)), y2
    return ('l', y, _subst(body, x, s))

def _step(t):
    """One leftmost-outermost β-step; None if already in normal form."""
    k = t[0]
    if k == 'a':
        f, a = t[1], t[2]
        if f[0] == 'l': return _subst(f[2], f[1], a)
        if (f2 := _step(f)) is not None: return ('a', f2, a)
        if (a2 := _step(a)) is not None: return ('a', f, a2)
    elif k == 'l':
        if (b2 := _step(t[2])) is not None: return ('l', t[1], b2)
    return None

def _normalize(t, max_steps=200):
    for _ in range(max_steps):
        n = _step(t)
        if n is None: return t
        t = n
    return None                                  # presumed diverging

def _pretty(t):
    if t[0] == 'v': return t[1]
    if t[0] == 'l': return f"(\\{t[1]}.{_pretty(t[2])})"
    return f"({_pretty(t[1])} {_pretty(t[2])})"

def _debruijn(t, env=()):
    """α-canonical form: bound vars → #i, free vars keep their name."""
    k = t[0]
    if k == 'v':
        return f"#{env.index(t[1])}" if t[1] in env else t[1]
    if k == 'l': return f"(\\.{_debruijn(t[2], (t[1],) + env)})"
    return f"({_debruijn(t[1], env)} {_debruijn(t[2], env)})"

_LAM_TOK = re.compile(r'[()\\.\u03bb]|[A-Za-z_]\w*')

def _parse_lam(s: str):
    toks, i = _LAM_TOK.findall(s), [0]
    def peek(): return toks[i[0]] if i[0] < len(toks) else None
    def pop():  t = peek(); i[0] += 1; return t
    def atom():
        t = peek()
        if t == '(':
            pop(); e = expr()
            if pop() != ')': raise ValueError("missing )")
            return e
        if t in ('\\', 'λ'):
            pop(); name = pop()
            if pop() != '.': raise ValueError("missing .")
            return ('l', name, expr())
        if t is None or t in (')', '.'):
            raise ValueError(f"unexpected {t!r}")
        return ('v', pop())
    def expr():
        e = atom()
        while peek() not in (None, ')'):
            e = ('a', e, atom())
        return e
    out = expr()
    if i[0] != len(toks): raise ValueError("trailing tokens")
    return out

_LAM_CONSTS = ['a', 'b', 'c', 'd']

def _gen_lam(depth, rng, env=()):
    if depth <= 0:
        if env and rng.random() < 0.7: return ('v', rng.choice(env))
        return ('v', rng.choice(_LAM_CONSTS))
    r = rng.random()
    if r < 0.35:
        name = f"v{len(env)}"
        return ('l', name, _gen_lam(depth - 1, rng, env + (name,)))
    if r < 0.9:
        return ('a', _gen_lam(depth - 1, rng, env),
                     _gen_lam(depth - 1, rng, env))
    return _gen_lam(0, rng, env)


@dataclass
class LambdaReductionConfig(Config):
    depth: int = 2

    def update(self, c=1):
        self.depth += c


class LambdaReduction(Task):
    def __init__(self, config=None):
        super().__init__(config=config or LambdaReductionConfig())

    def generate(self):
        rng = random.Random()
        for _ in range(300):
            t = _gen_lam(self.config.depth, rng)
            if _step(t) is None: continue        # no redex → boring
            nf = _normalize(t)
            if nf is None: continue              # diverged
            if len(_pretty(nf)) > 250: continue  # size blowup
            return Problem(
                metadata=edict(term=_pretty(t), normal_form=_pretty(nf)),
                answer=_pretty(nf),
            )
        raise RuntimeError("could not sample a non-trivial terminating term")

    def prompt(self, metadata):
        return (
            "Reduce the following untyped λ-term to β-normal form.\n"
            "Syntax: `\\x.body` denotes λx.body; application is left-associative "
            "juxtaposition; free identifiers are treated as constants.\n\n"
            f"Term: {metadata['term']}\n\n"
            "The answer is the β-normal form (compared up to α-equivalence)."
        )

    def score_answer(self, answer, entry):
        if answer is None: return 0.0
        try:
            got = _parse_lam(str(answer).strip())
        except Exception:
            return 0.0
        ref = _parse_lam(entry.answer)
        return float(_debruijn(got) == _debruijn(ref))


# ─────────────────────────────────────────────────────────────────────────────
# First-order term unification
# ─────────────────────────────────────────────────────────────────────────────
# Prolog-ish syntax:  f(a, X, g(Y, b))  —  Upper = Var, lower = functor/atom.

_TERM_TOK = re.compile(r'[A-Za-z_]\w*|[(),]')

def _show_term(t) -> str:
    if isinstance(t, Var): return t.token
    if isinstance(t, tuple):
        return f"{t[0]}({','.join(_show_term(x) for x in t[1:])})"
    return str(t)

def _has_var(t) -> bool:
    if isinstance(t, Var): return True
    if isinstance(t, tuple): return any(_has_var(x) for x in t[1:])
    return False

def _vars_in(t):
    if isinstance(t, Var): return {t}
    if isinstance(t, tuple):
        s = set()
        for x in t[1:]: s |= _vars_in(x)
        return s
    return set()

_FUNCS = ['f', 'g', 'h', 'p', 'q']
_ATOMS = ['a', 'b', 'c', 'd', 'e']

def _gen_term(depth, rng, var_pool):
    if depth <= 0 or rng.random() < 0.3:
        if var_pool and rng.random() < 0.5:
            return rng.choice(var_pool)
        return rng.choice(_ATOMS)
    head = rng.choice(_FUNCS)
    arity = rng.randint(1, 3)
    return (head, *[_gen_term(depth - 1, rng, var_pool) for _ in range(arity)])


@dataclass
class TermUnificationConfig(Config):
    depth: int = 2
    n_vars: int = 2

    def update(self, c=1):
        self.depth += c
        self.n_vars += c


class TermUnification(Task):
    def __init__(self, config=None):
        super().__init__(config=config or TermUnificationConfig())

    def generate(self):
        rng = random.Random()
        NAMES = ['X', 'Y', 'Z', 'W', 'U', 'V', 'X1', 'Y1', 'Z1', 'W1']
        for _ in range(300):
            n = max(2, min(int(self.config.n_vars), len(NAMES)))
            pool = [mkvar(nm) for nm in NAMES[:n]]
            skel = _gen_term(self.config.depth + 1, rng, pool)
            if len(_vars_in(skel)) < 2: continue

            def partial():
                keep = [v for v in _vars_in(skel) if rng.random() < 0.5]
                sub = {v: _gen_term(self.config.depth, rng, []) for v in keep}
                return reify(skel, sub)

            t1, t2 = partial(), partial()
            s1, s2 = _show_term(t1), _show_term(t2)
            if s1 == s2: continue
            mgu = unify(t1, t2)
            if not mgu: continue                  # False or empty
            fully = {k.token: reify(k, mgu) for k in mgu}
            if any(_has_var(v) for v in fully.values()): continue
            nice = {k: _show_term(v) for k, v in fully.items()}
            return Problem(
                metadata=edict(term1=s1, term2=s2, mgu=nice),
                answer=repr(dict(sorted(nice.items()))),
            )
        raise RuntimeError("could not sample a unifiable pair")

    def prompt(self, metadata):
        return (
            "Find the most general unifier (MGU) of the following first-order "
            "terms.\nUppercase identifiers are variables; lowercase are "
            "constants / function symbols.\n\n"
            f"T1 = {metadata['term1']}\n"
            f"T2 = {metadata['term2']}\n\n"
            "The answer is a Python dict mapping each bound variable (as a "
            "string key) to its fully-resolved ground term (as a string "
            "value), with keys sorted alphabetically.\n"
            "Example: {'X': 'f(a)', 'Y': 'b'}"
        )

    def score_answer(self, answer, entry):
        if answer is None: return 0.0
        try:
            got = ast.literal_eval(str(answer).strip())
            if not isinstance(got, dict): return 0.0
            got = {str(k): str(v).replace(' ', '') for k, v in got.items()}
        except Exception:
            return 0.0
        ref = {k: v.replace(' ', '') for k, v in entry.metadata['mgu'].items()}
        return float(got == ref)