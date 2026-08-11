"""Symbolic rewriting: λ-calculus β-reduction.

- `lambda_reduction` — reduce an untyped λ-term to β-normal form.
  Instances mix arbitrary λ-term sampling with *anti-reduction*: sample an NF,
  then insert redexes that reduce back to it. Three elementary moves:
      1. dummy        (λx.M) N          x ∉ FV(M)
      2. identity     (λx.x) M
      3. substitution (λx.M[T↦x]) T     (optionally duplicating T)
  Move 3 can fail capture-avoidance silently; we catch it by normalising the
  result and checking α-equivalence with the target NF.
"""
from dataclasses import dataclass
from collections import Counter
import ast, hashlib, random, re

from reasoning_core.template import Task, Entry, Config, render_payload, edict


# ─────────────────────────────────────────────────────────────────────────────
# λ-calculus core
# ─────────────────────────────────────────────────────────────────────────────
# term ::= ('v', name) | ('l', name, body) | ('a', fun, arg)

def _fv(t):
    k = t[0]
    if k == 'v': return {t[1]}
    if k == 'l': return _fv(t[2]) - {t[1]}
    return _fv(t[1]) | _fv(t[2])

def _all_names(t):
    k = t[0]
    if k == 'v': return {t[1]}
    if k == 'l': return _all_names(t[2]) | {t[1]}
    return _all_names(t[1]) | _all_names(t[2])

def _fresh(avoid):
    i = 0
    while (n := f"_{i}") in avoid: i += 1
    return n

def _subst(t, x, s):
    k = t[0]
    if k == 'v': return s if t[1] == x else t
    if k == 'a': return ('a', _subst(t[1], x, s), _subst(t[2], x, s))
    y, body = t[1], t[2]
    if y == x: return t
    if x not in _fv(body): return t
    if y in _fv(s):
        y2 = _fresh(_fv(body) | _fv(s) | {x})
        body, y = _subst(body, y, ('v', y2)), y2
    return ('l', y, _subst(body, x, s))

def _step(t):
    """One leftmost-outermost β-step; None if already in NF."""
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
    return None

def _term_size(t):
    if t[0] == 'v': return 1
    if t[0] == 'l': return 1 + _term_size(t[2])
    return 1 + _term_size(t[1]) + _term_size(t[2])

def _normalize_trace(t, max_steps=500):
    out = [t]
    for _ in range(max_steps):
        n = _step(t)
        if n is None: return out
        t = n
        out.append(t)
    return None

def _safe_normalize_trace(t, max_steps=500, max_size=2000):
    out = [t]
    try:
        for _ in range(max_steps):
            if _term_size(t) > max_size: return None
            n = _step(t)
            if n is None: return out
            t = n
            out.append(t)
    except RecursionError:
        return None
    return None

def _pretty(t):
    if t[0] == 'v': return t[1]
    if t[0] == 'l': return f"(\\{t[1]}.{_pretty(t[2])})"
    return f"({_pretty(t[1])} {_pretty(t[2])})"

def _debruijn(t, env=()):
    k = t[0]
    if k == 'v':
        return f"#{env.index(t[1])}" if t[1] in env else t[1]
    if k == 'l': return f"(\\.{_debruijn(t[2], (t[1],) + env)})"
    return f"({_debruijn(t[1], env)} {_debruijn(t[2], env)})"

def _alpha_normalize(t, env=None, state=None):
    """Give bound variables deterministic names without changing free variables."""
    env = {} if env is None else env
    state = {'next': 0, 'free': _fv(t)} if state is None else state
    if t[0] == 'v':
        return ('v', env.get(t[1], t[1]))
    if t[0] == 'a':
        return ('a', _alpha_normalize(t[1], env, state),
                     _alpha_normalize(t[2], env, state))
    while (new := f"x{state['next']}") in state['free']:
        state['next'] += 1
    state['next'] += 1
    return ('l', new, _alpha_normalize(t[2], env | {t[1]: new}, state))

def _skeleton(t):
    if t[0] == 'v': return 'v'
    if t[0] == 'l': return ('l', _skeleton(t[2]))
    return ('a', _skeleton(t[1]), _skeleton(t[2]))

def _has_shadowing(t, bound=()):
    if t[0] == 'l':
        return t[1] in bound or _has_shadowing(t[2], bound + (t[1],))
    if t[0] == 'a':
        return _has_shadowing(t[1], bound) or _has_shadowing(t[2], bound)
    return False

def _shadowing_count(t, bound=()):
    if t[0] == 'l':
        return int(t[1] in bound) + _shadowing_count(t[2], bound + (t[1],))
    if t[0] == 'a':
        return _shadowing_count(t[1], bound) + _shadowing_count(t[2], bound)
    return 0

def _var_count(t, x):
    if t[0] == 'v': return int(t[1] == x)
    if t[0] == 'l': return 0 if t[1] == x else _var_count(t[2], x)
    return _var_count(t[1], x) + _var_count(t[2], x)

def _redexes(t, path=()):
    if t[0] == 'a':
        if t[1][0] == 'l': yield path, t
        yield from _redexes(t[1], path + (1,))
        yield from _redexes(t[2], path + (2,))
    elif t[0] == 'l':
        yield from _redexes(t[2], path + (2,))

def _lo_redex(t, path=()):
    if t[0] == 'a':
        if t[1][0] == 'l': return path, t
        out = _lo_redex(t[1], path + (1,))
        return out if out is not None else _lo_redex(t[2], path + (2,))
    if t[0] == 'l': return _lo_redex(t[2], path + (2,))
    return None

def _needs_alpha(body, x, arg):
    if body[0] == 'v': return False
    if body[0] == 'a':
        return _needs_alpha(body[1], x, arg) or _needs_alpha(body[2], x, arg)
    y, b = body[1], body[2]
    if y == x: return False
    return (y in _fv(arg) and x in _fv(b)) or _needs_alpha(b, x, arg)

def _alpha_renaming_redexes(t):
    return sum(_needs_alpha(r[1][2], r[1][1], r[2]) for _, r in _redexes(t))

def _capture_risk_redexes(t):
    return _alpha_renaming_redexes(t)

_BUCKETS = ('identity', 'dummy', 'substitution', 'duplication', 'shadowing',
            'alpha-renaming', 'deep-redex', 'root-redex', 'arbitrary')

def _phenomena(t):
    out = set()
    if _has_shadowing(t): out.add('shadowing')
    if _alpha_renaming_redexes(t): out.add('alpha-renaming')
    for path, r in _redexes(t):
        x, body = r[1][1], r[1][2]
        if not path: out.add('root-redex')
        else: out.add('deep-redex')
        if body == ('v', x): out.add('identity')
        elif x not in _fv(body): out.add('dummy')
        else:
            out.add('substitution')
            if _var_count(body, x) > 1: out.add('duplication')
    return out

def _contracted_phenomena(trace):
    out, counts = set(), Counter()
    for t in trace[:-1]:
        got = _lo_redex(t)
        if got is None: continue
        path, r = got
        x, body, arg = r[1][1], r[1][2], r[2]
        b = 'root-redex' if not path else 'deep-redex'
        out.add(b); counts[b] += 1
        if body == ('v', x):
            out.add('identity'); counts['identity'] += 1
        elif x not in _fv(body):
            out.add('dummy'); counts['dummy'] += 1
        else:
            out.add('substitution'); counts['substitution'] += 1
            if _var_count(body, x) > 1:
                out.add('duplication'); counts['duplication'] += 1
        if _needs_alpha(body, x, arg):
            out.add('alpha-renaming'); counts['alpha-renaming'] += 1
    return out, counts

def _split_key(t, nf):
    s = repr((_debruijn(t), _debruijn(nf))).encode()
    return hashlib.sha1(s).hexdigest()[:16]

_LAM_TOK = re.compile(r'[()\\.\u03bb]|[A-Za-z_]\w*')

def _strict_tokens(s, token_re):
    """Tokenize while requiring every non-whitespace character to be valid."""
    out, end = [], 0
    for match in token_re.finditer(s):
        if s[end:match.start()].strip():
            raise ValueError(f"invalid input near {s[end:match.start()]!r}")
        out.append(match.group())
        end = match.end()
    if s[end:].strip():
        raise ValueError(f"invalid trailing input {s[end:]!r}")
    return out

def _parse_lam(s: str):
    toks, i = _strict_tokens(s, _LAM_TOK), [0]
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
            if name is None or not re.fullmatch(r'[A-Za-z_]\w*', name):
                raise ValueError("binder must be an identifier")
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


# ─────────────────────────────────────────────────────────────────────────────
# Anti-reduction: sample NF, then insert redexes that reduce back to it
# ─────────────────────────────────────────────────────────────────────────────
_LAM_CONSTS = ['a', 'b', 'c', 'd']

def _gen_nf(depth, rng, env):
    """Sample a β-normal term.
        NF       ::=  λx.NF  |  neutral
        neutral  ::=  (var|const) NF*          (head is never a λ)
    """
    if depth > 0 and rng.random() < 0.35:
        name = rng.choice(env) if env and rng.random() < 0.25 else f"v{len(env)}"
        return ('l', name, _gen_nf(depth - 1, rng, env + (name,)))
    head = ('v', rng.choice(env) if env and rng.random() < 0.7
                  else rng.choice(_LAM_CONSTS))
    n_args = 0 if depth <= 0 else rng.randint(0, 2)
    t = head
    for _ in range(n_args):
        t = ('a', t, _gen_nf(depth - 1, rng, env))
    return t

def _gen_term(depth, rng, env):
    if depth <= 0 or rng.random() < 0.25:
        return ('v', rng.choice(env) if env and rng.random() < 0.7 else rng.choice(_LAM_CONSTS))
    if depth > 2 and rng.random() < 0.25:
        x, y = f"v{len(env)}", f"v{len(env) + 1}"
        body = ('l', y, ('l', y, ('a', ('v', x), _gen_term(depth - 3, rng, env + (x, y, y)))))
        return ('a', ('l', x, body), ('v', y))
    if rng.random() < 0.35:
        name = rng.choice(env) if env and rng.random() < 0.3 else f"v{len(env)}"
        return ('l', name, _gen_term(depth - 1, rng, env + (name,)))
    if rng.random() < 0.25:
        x = f"v{len(env)}"
        return ('a', ('l', x, _gen_term(depth - 1, rng, env + (x,))),
                     _gen_term(depth - 1, rng, env))
    return ('a', _gen_term(depth - 1, rng, env), _gen_term(depth - 1, rng, env))

def _positions(t, path=()):
    yield path, t
    if t[0] == 'a':
        yield from _positions(t[1], path + (1,))
        yield from _positions(t[2], path + (2,))
    elif t[0] == 'l':
        yield from _positions(t[2], path + (2,))

def _replace_at(t, path, new):
    if not path: return new
    i, rest = path[0], path[1:]
    if t[0] == 'a':
        return (('a', _replace_at(t[1], rest, new), t[2]) if i == 1
                else ('a', t[1], _replace_at(t[2], rest, new)))
    return ('l', t[1], _replace_at(t[2], rest, new))

def _replace_all(t, target, new):
    if t == target: return new
    if t[0] == 'a':
        return ('a', _replace_all(t[1], target, new),
                     _replace_all(t[2], target, new))
    if t[0] == 'l':
        return ('l', t[1], _replace_all(t[2], target, new))
    return t

def _insert_redex(M, rng, arg_depth):
    """One anti-reduction move at a random subterm of M."""
    path, S = rng.choice(list(_positions(M)))
    x = _fresh(_all_names(M))
    inner = [(p, s) for p, s in _positions(S) if p]
    r = rng.random()

    if r < 0.2 or not inner:                        # dummy:    (λx.S) N
        N = _gen_nf(rng.randint(0, min(2, arg_depth)), rng, ())
        new = ('a', ('l', x, S), N)
    elif r < 0.35:                                  # identity: (λx.x) S
        new = ('a', ('l', x, ('v', x)), S)
    else:                                           # subst:    (λx.S[T↦x]) T
        _, T = rng.choice(inner)
        hits = [p for p, s in inner if s == T]
        S_abs = (_replace_all(S, T, ('v', x)) if rng.random() < 0.5
                 else _replace_at(S, rng.choice(hits), ('v', x)))
        new = ('a', ('l', x, S_abs), T)

    return _replace_at(M, path, new)

def _sample_by_antireduction(cfg, rng):
    nf = _gen_nf(cfg.nf_depth, rng, ())
    t = nf
    for _ in range(cfg.n_insertions):
        t2 = _insert_redex(t, rng, cfg.nf_depth)
        if len(_pretty(t2)) > cfg.max_chars: break
        t = t2
    return t, nf


@dataclass
class LambdaReductionConfig(Config):
    nf_depth: int = 5
    term_depth: int = 5
    n_insertions: int = 6
    min_steps: int = 5
    max_steps: int = 500
    max_chars: int = 500
    min_alpha_renaming: int = 1
    min_shadowing: int = 1
    anti_reduction_prob: float = 0.6

    def apply_difficulty(self, level):
        self.nf_depth += level
        self.term_depth += level
        self.n_insertions += level
        self.min_steps += level


class LambdaReduction(Task):
    summary = "Reduce lambda calculus terms to normal form with renaming and shadowing."
    def __init__(self, config=None):
        super().__init__(config=config or LambdaReductionConfig())

    def generate_entry(self):
        rng = random.Random()
        cfg = self.config
        source = 'anti_reduction' if rng.random() < cfg.anti_reduction_prob else 'arbitrary'
        target = rng.choice(_BUCKETS[:-1]) if source == 'anti_reduction' else 'arbitrary'
        for _ in range(3000):
            if source == 'anti_reduction':
                t, nf = _sample_by_antireduction(cfg, rng)
            else:
                t = _gen_term(cfg.term_depth, rng, ())

            if _term_size(t) > cfg.max_chars * 4: continue
            term = _pretty(t)
            if len(term) > cfg.max_chars: continue

            trace = _safe_normalize_trace(t, cfg.max_steps, cfg.max_chars * 4)
            if trace is None: continue
            if len(trace) - 1 < cfg.min_steps: continue
            if source == 'anti_reduction' and _debruijn(trace[-1]) != _debruijn(nf): continue
            nf = _alpha_normalize(trace[-1])
            normal = _pretty(nf)
            if not (3 <= len(normal) <= cfg.max_chars): continue
            if _step(nf) is not None: continue
            shadowing = _shadowing_count(t)
            syntactic_alpha_renaming = _alpha_renaming_redexes(t)
            syntactic_buckets = _phenomena(t)
            trace_buckets, trace_bucket_counts = _contracted_phenomena(trace)
            if shadowing:
                trace_buckets.add('shadowing')
            buckets = trace_buckets
            if target == 'shadowing' and shadowing < cfg.min_shadowing: continue
            if target == 'alpha-renaming' and trace_bucket_counts.get('alpha-renaming', 0) < cfg.min_alpha_renaming: continue
            if target != 'arbitrary' and target not in buckets: continue

            return Entry(
                metadata=edict(
                    term=term,
                    normal_form=normal,
                    beta_steps=len(trace) - 1,
                    has_shadowing=_has_shadowing(t),
                    shadowing=shadowing,
                    capture_risk=trace_bucket_counts.get('alpha-renaming', 0),
                    alpha_renaming=trace_bucket_counts.get('alpha-renaming', 0),
                    syntactic_alpha_renaming=syntactic_alpha_renaming,
                    skeleton=_skeleton(t),
                    nf_skeleton=_skeleton(nf),
                    split_key=_split_key(t, nf),
                    generator=source,
                    buckets=sorted(buckets),
                    syntactic_buckets=sorted(syntactic_buckets),
                    trace_bucket_counts=dict(trace_bucket_counts),
                    target_bucket=target,
                ),
                answer=normal,
            )
        raise RuntimeError("could not sample a valid λ-term")

    def render_prompt(self, metadata):
        return (
            "Reduce the following untyped λ-term to β-normal form.\n"
            "Syntax: `\\x.body` is λx.body; juxtaposition is left-associative application; free identifiers are constants.\n\n"
            f"Term: {metadata['term']}\n\n"
            "The answer is the β-normal form (compared up to α-equivalence)."
        )

    def deduplication_key(self, problem):
        return problem.metadata["split_key"]

    def score_answer(self, answer, entry):
        if answer is None: return 0.0
        try:
            got = _parse_lam(str(answer).strip())
        except Exception:
            return 0.0
        ref = _parse_lam(entry.answer)
        if _debruijn(got) == _debruijn(ref):
            return 1.0
        trace = _safe_normalize_trace(
            got,
            max_steps=max(500, int(entry.metadata.get("beta_steps", 0)) * 4),
            max_size=max(2000, len(str(answer)) * 20),
        )
        if trace is not None and _debruijn(trace[-1]) == _debruijn(ref):
            return 0.8
        return 0.0


def lambda_reduction_shortcut_report(train, test, predictions=None):
    """Diagnostics for splits and shortcut baselines; predictions are test-order or term-keyed."""
    train, test = list(train), list(test)
    def md(e): return e.metadata if hasattr(e, 'metadata') else edict(e['metadata'])
    def ans(e): return e.answer if hasattr(e, 'answer') else e['answer']
    def term(e): return _parse_lam(md(e).term)
    def sk(e, k):
        return repr(md(e).get(k) or _skeleton(term(e)))
    def score(p, e):
        try: return float(_debruijn(_parse_lam(str(p).strip())) == _debruijn(_parse_lam(ans(e))))
        except Exception: return 0.0
    def pred_for(i, e):
        if predictions is None: return None
        return predictions.get(md(e).term) if isinstance(predictions, dict) else predictions[i]
    def acc(items, f):
        items = list(items)
        return None if not items else sum(f(i, e) for i, e in items) / len(items)
    def root_only(t):
        for _ in range(500):
            if t[0] != 'a' or t[1][0] != 'l': return t
            t = _subst(t[1][2], t[1][1], t[2])
        return t
    def delete_wrappers(t):
        if t[0] == 'a' and t[1][0] == 'l' and t[1][1] not in _fv(t[1][2]):
            return delete_wrappers(t[1][2])
        if t[0] == 'a': return ('a', delete_wrappers(t[1]), delete_wrappers(t[2]))
        if t[0] == 'l': return ('l', t[1], delete_wrappers(t[2]))
        return t

    train_pairs = {(md(e).term, ans(e)) for e in train}
    train_in_skel = {sk(e, 'skeleton') for e in train}
    train_nf_skel = {sk(e, 'nf_skeleton') for e in train}
    all_items = list(train) + list(test)
    source_counts = Counter(md(e).get('generator', 'unknown') for e in all_items)
    indexed = list(enumerate(test))
    lm_acc = None if predictions is None else acc(indexed, lambda i, e: score(pred_for(i, e), e))
    by_steps = {}
    if predictions is not None:
        for s in sorted({md(e).beta_steps for e in test if 'beta_steps' in md(e)}):
            by_steps[str(s)] = acc(((i, e) for i, e in indexed if md(e).get('beta_steps') == s),
                                   lambda i, e: score(pred_for(i, e), e))
    return edict(
        exact_train_test_duplicate_rate=sum((md(e).term, ans(e)) in train_pairs for e in test) / len(test),
        input_skeleton_overlap=sum(sk(e, 'skeleton') in train_in_skel for e in test) / len(test),
        normal_form_skeleton_overlap=sum(sk(e, 'nf_skeleton') in train_nf_skel for e in test) / len(test),
        accepted_generator_ratio={k: v / len(all_items) for k, v in source_counts.items()},
        one_step_contraction_baseline=acc(indexed, lambda i, e: score(_pretty(_step(term(e)) or term(e)), e)),
        root_redex_only_baseline=acc(indexed, lambda i, e: score(_pretty(root_only(term(e))), e)),
        delete_wrapper_baseline=acc(indexed, lambda i, e: score(_pretty(delete_wrappers(term(e))), e)),
        tuned_accuracy=lm_acc,
        tuned_accuracy_by_beta_steps=by_steps or None,
        tuned_capture_risk_accuracy=None if predictions is None else acc(
            ((i, e) for i, e in indexed if md(e).get('capture_risk', 0)),
            lambda i, e: score(pred_for(i, e), e)),
        tuned_ood_steps_accuracy=None if predictions is None else acc(
            ((i, e) for i, e in indexed if md(e).get('beta_steps', 0) >= 8),
            lambda i, e: score(pred_for(i, e), e)),
        tuned_unseen_skeleton_accuracy=None if predictions is None else acc(
            ((i, e) for i, e in indexed if sk(e, 'skeleton') not in train_in_skel),
            lambda i, e: score(pred_for(i, e), e)),
        tuned_arbitrary_generator_accuracy=None if predictions is None else acc(
            ((i, e) for i, e in indexed if md(e).get('generator') == 'arbitrary'),
            lambda i, e: score(pred_for(i, e), e)),
        tuned_alpha_renaming_required_accuracy=None if predictions is None else acc(
            ((i, e) for i, e in indexed if md(e).get('alpha_renaming', md(e).get('capture_risk', 0))),
            lambda i, e: score(pred_for(i, e), e)),
        tuned_duplication_accuracy=None if predictions is None else acc(
            ((i, e) for i, e in indexed if 'duplication' in md(e).get('buckets', ())),
            lambda i, e: score(pred_for(i, e), e)),
    )








@dataclass(frozen=True)
class Rule:
    name: str
    lhs: object
    rhs: object


def _rw_isvar(t):
    return isinstance(t, str) and t[:1].isupper()


def _rw_show(t):
    if isinstance(t, tuple):
        return f"{t[0]}({','.join(_rw_show(x) for x in t[1:])})"
    return str(t)


def _rw_vars(t):
    if _rw_isvar(t):
        return {t}
    if isinstance(t, tuple):
        return set().union(*(_rw_vars(x) for x in t[1:])) if len(t) > 1 else set()
    return set()


def _rw_match(p, t, env=None):
    env = {} if env is None else dict(env)

    if _rw_isvar(p):
        if p in env:
            return env if env[p] == t else None
        env[p] = t
        return env

    if isinstance(p, tuple):
        if not isinstance(t, tuple) or p[0] != t[0] or len(p) != len(t):
            return None
        for a, b in zip(p[1:], t[1:]):
            env = _rw_match(a, b, env)
            if env is None:
                return None
        return env

    return env if p == t else None


def _rw_subst(t, env):
    if _rw_isvar(t):
        return env[t]
    if isinstance(t, tuple):
        return (t[0], *[_rw_subst(x, env) for x in t[1:]])
    return t


def _rw_positions(t, path=()):
    yield path, t
    if isinstance(t, tuple):
        for i, x in enumerate(t[1:], 1):
            yield from _rw_positions(x, path + (i,))


def _rw_replace(t, path, new):
    if not path:
        return new
    i, rest = path[0], path[1:]
    args = list(t[1:])
    args[i - 1] = _rw_replace(args[i - 1], rest, new)
    return (t[0], *args)


def _rw_step(t, rules):
    for r in rules:
        env = _rw_match(r.lhs, t)
        if env is not None:
            return _rw_subst(r.rhs, env), r.name

    if isinstance(t, tuple):
        for i, x in enumerate(t[1:], 1):
            out = _rw_step(x, rules)
            if out is not None:
                x2, name = out
                args = list(t[1:])
                args[i - 1] = x2
                return (t[0], *args), name

    return None


def _rw_step_rule_major(t, rules):
    """Diagnostic alternative: rule priority before position priority."""
    for r in rules:
        for path, subterm in _rw_positions(t):
            env = _rw_match(r.lhs, subterm)
            if env is not None:
                return _rw_replace(t, path, _rw_subst(r.rhs, env)), r.name
    return None


def _rw_trace_with(t, rules, step, max_steps=200):
    terms, names = [t], []
    for _ in range(max_steps):
        out = step(t, rules)
        if out is None:
            return terms, names
        t, name = out
        terms.append(t)
        names.append(name)
    return None, names


def _rw_trace(t, rules, max_steps=200):
    terms, names = [t], []
    for _ in range(max_steps):
        out = _rw_step(t, rules)
        if out is None:
            return terms, names
        t, name = out
        terms.append(t)
        names.append(name)
    return None, names


def _rw_normalize(t, rules, max_steps=200):
    terms, names = _rw_trace(t, rules, max_steps)
    if terms is None:
        return None, names
    return terms[-1], names


def _rw_normalize_with(t, rules, step, max_steps=200):
    terms, names = _rw_trace_with(t, rules, step, max_steps)
    return (None, names) if terms is None else (terms[-1], names)


def _rw_is_subterm(target, term, proper=False):
    return any(sub == target and (not proper or path) for path, sub in _rw_positions(term))


def _rw_step_stats(t, rules):
    """Depth and overlap count at the position selected by position-major order."""
    for path, subterm in _rw_positions(t):
        matches = [r for r in rules if _rw_match(r.lhs, subterm) is not None]
        if matches:
            return len(path), len(matches)
    return 0, 0


def _rw_split_key(rules, term, normal_form):
    """Canonicalize an instance up to arity-preserving symbol renaming."""
    variables, symbols = {}, {}
    counters = Counter()

    def canon(t):
        if _rw_isvar(t):
            variables.setdefault(t, f"V{len(variables)}")
            return variables[t]
        if isinstance(t, tuple):
            key = (t[0], len(t) - 1)
            if key not in symbols:
                arity = key[1]
                symbols[key] = f"f{arity}_{counters[arity]}"
                counters[arity] += 1
            return (symbols[key], *(canon(x) for x in t[1:]))
        key = (t, 0)
        if key not in symbols:
            symbols[key] = f"c{counters[0]}"
            counters[0] += 1
        return symbols[key]

    value = (tuple((canon(r.lhs), canon(r.rhs)) for r in rules),
             canon(term), canon(normal_form))
    return hashlib.sha1(repr(value).encode()).hexdigest()[:16]


_RW_TOK = re.compile(r'[A-Za-z_]\w*|\d+|[(),]')


def _rw_parse(s):
    toks, i = _strict_tokens(s, _RW_TOK), [0]

    def peek():
        return toks[i[0]] if i[0] < len(toks) else None

    def pop():
        t = peek()
        i[0] += 1
        return t

    def term():
        name = pop()
        if name is None or name in '(),':
            raise ValueError("bad term")
        if peek() != '(':
            return name

        pop()
        args = []
        if peek() != ')':
            while True:
                args.append(term())
                if peek() != ',':
                    break
                pop()

        if pop() != ')':
            raise ValueError("missing )")

        return (name, *args)

    out = term()
    if i[0] != len(toks):
        raise ValueError("trailing tokens")
    return out


def _rw_gen_ground(depth, rng, consts, heads):
    if depth <= 0 or rng.random() < 0.42:
        return rng.choice(consts)
    f, arity = rng.choice(heads)
    return (f, *[_rw_gen_ground(depth - 1, rng, consts, heads) for _ in range(arity)])


def _rw_fill_env(vars_, env, rng, consts, heads, depth, rules):
    env = dict(env)
    for v in sorted(vars_):
        if v in env:
            continue
        t = _rw_gen_ground(rng.randint(0, depth), rng, consts, heads)
        t, _ = _rw_normalize(t, rules)
        if t is None:
            return None
        env[v] = t
    return env


def _rw_insert_redex(t, rules, rng, consts, heads, depth):
    positions = list(_rw_positions(t))
    rng.shuffle(positions)

    rs = list(rules)
    rng.shuffle(rs)

    for path, sub in positions:
        for r in rs:
            env = _rw_match(r.rhs, sub)
            if env is None:
                continue

            env = _rw_fill_env(_rw_vars(r.lhs), env, rng, consts, heads, depth, rules)
            if env is None:
                continue

            redex = _rw_subst(r.lhs, env)
            if redex != sub:
                return _rw_replace(t, path, redex), r.name

    return None, None


def _rw_rule_text(rules):
    return '\n'.join(f"- {_rw_show(r.lhs)} -> {_rw_show(r.rhs)}" for r in rules)


def _rw_pack_arith():
    X, Y, Z = 'X', 'Y', 'Z'
    return edict(
        name='arith',
        consts=['0', '1', '2', 'a', 'b', 'c'],
        heads=[('add', 2), ('mul', 2), ('sub', 2), ('neg', 1), ('pow', 2)],
        rules=[
            Rule('add_zero_l', ('add', '0', X), X),
            Rule('add_zero_r', ('add', X, '0'), X),
            Rule('mul_one_l',  ('mul', '1', X), X),
            Rule('mul_one_r',  ('mul', X, '1'), X),
            Rule('mul_zero_l', ('mul', '0', X), '0'),
            Rule('mul_zero_r', ('mul', X, '0'), '0'),
            Rule('sub_zero',   ('sub', X, '0'), X),
            Rule('sub_self',   ('sub', X, X), '0'),
            Rule('neg_neg',    ('neg', ('neg', X)), X),
            Rule('pow_one',    ('pow', X, '1'), X),
            Rule('pow_zero',   ('pow', X, '0'), '1'),
            Rule('factor_l',   ('add', ('mul', X, Y), ('mul', X, Z)), ('mul', X, ('add', Y, Z))),
        ],
    )


def _rw_pack_bool():
    X, Y = 'X', 'Y'
    return edict(
        name='bool',
        consts=['true', 'false', 'a', 'b', 'c'],
        heads=[('and', 2), ('or', 2), ('not', 1), ('if', 3), ('eq', 2)],
        rules=[
            Rule('and_true_l',  ('and', 'true', X), X),
            Rule('and_true_r',  ('and', X, 'true'), X),
            Rule('and_false_l', ('and', 'false', X), 'false'),
            Rule('and_false_r', ('and', X, 'false'), 'false'),
            Rule('and_idem',    ('and', X, X), X),
            Rule('or_false_l',  ('or', 'false', X), X),
            Rule('or_false_r',  ('or', X, 'false'), X),
            Rule('or_true_l',   ('or', 'true', X), 'true'),
            Rule('or_true_r',   ('or', X, 'true'), 'true'),
            Rule('or_idem',     ('or', X, X), X),
            Rule('not_not',     ('not', ('not', X)), X),
            Rule('if_true',     ('if', 'true', X, Y), X),
            Rule('if_false',    ('if', 'false', X, Y), Y),
            Rule('eq_self',     ('eq', X, X), 'true'),
        ],
    )


def _rw_pack_list():
    X, XS, YS = 'X', 'XS', 'YS'
    return edict(
        name='list',
        consts=['nil', 'a', 'b', 'c', '0'],
        heads=[('cons', 2), ('append', 2), ('head', 1), ('tail', 1), ('len', 1), ('s', 1)],
        rules=[
            Rule('append_nil',  ('append', 'nil', XS), XS),
            Rule('append_cons', ('append', ('cons', X, XS), YS), ('cons', X, ('append', XS, YS))),
            Rule('head_cons',   ('head', ('cons', X, XS)), X),
            Rule('tail_cons',   ('tail', ('cons', X, XS)), XS),
            Rule('len_nil',     ('len', 'nil'), '0'),
            Rule('len_cons',    ('len', ('cons', X, XS)), ('s', ('len', XS))),
        ],
    )


def _rw_pack_ast():
    X, Y = 'X', 'Y'
    return edict(
        name='ast',
        consts=['unit', 'true', 'false', 'a', 'b', 'c'],
        heads=[('pair', 2), ('fst', 1), ('snd', 1), ('id', 1), ('const', 2), ('if', 3), ('let', 2)],
        rules=[
            Rule('fst_pair', ('fst', ('pair', X, Y)), X),
            Rule('snd_pair', ('snd', ('pair', X, Y)), Y),
            Rule('id',       ('id', X), X),
            Rule('const',    ('const', X, Y), X),
            Rule('if_true',  ('if', 'true', X, Y), X),
            Rule('if_false', ('if', 'false', X, Y), Y),
            Rule('let_unit', ('let', 'unit', X), X),
        ],
    )


def _rw_pack_string():
    X, Y, Z = 'X', 'Y', 'Z'
    return edict(
        name='string',
        consts=['eps', 'a', 'b', 'c'],
        heads=[('cat', 2), ('rev', 1), ('quote', 1), ('lower', 1), ('upper', 1)],
        rules=[
            Rule('cat_eps_l',   ('cat', 'eps', X), X),
            Rule('cat_eps_r',   ('cat', X, 'eps'), X),
            Rule('cat_assoc',   ('cat', ('cat', X, Y), Z), ('cat', X, ('cat', Y, Z))),
            Rule('rev_eps',     ('rev', 'eps'), 'eps'),
            Rule('rev_rev',     ('rev', ('rev', X)), X),
            Rule('quote_quote', ('quote', ('quote', X)), ('quote', X)),
            Rule('lower_lower', ('lower', ('lower', X)), ('lower', X)),
            Rule('upper_upper', ('upper', ('upper', X)), ('upper', X)),
        ],
    )


def _rw_pack_logic():
    X, Y, Z = 'X', 'Y', 'Z'
    return edict(
        name='logic',
        consts=['true', 'false', 'p', 'q', 'r'],
        heads=[('imp', 2), ('iff', 2), ('xor', 2), ('not', 1), ('and', 2), ('or', 2)],
        rules=[
            Rule('imp_true_l',  ('imp', 'true', X), X),
            Rule('imp_false_l', ('imp', 'false', X), 'true'),
            Rule('imp_true_r',  ('imp', X, 'true'), 'true'),
            Rule('imp_false_r', ('imp', X, 'false'), ('not', X)),
            Rule('iff_self',    ('iff', X, X), 'true'),
            Rule('xor_self',    ('xor', X, X), 'false'),
            Rule('xor_false_l', ('xor', 'false', X), X),
            Rule('xor_false_r', ('xor', X, 'false'), X),
            Rule('xor_true_l',  ('xor', 'true', X), ('not', X)),
            Rule('xor_true_r',  ('xor', X, 'true'), ('not', X)),
            Rule('not_not',     ('not', ('not', X)), X),
            Rule('demorgan',    ('not', ('and', X, Y)), ('or', ('not', X), ('not', Y))),
        ],
    )


def _rw_pack_path():
    X, Y, Z = 'X', 'Y', 'Z'
    return edict(
        name='path',
        consts=['root', 'home', 'tmp', 'a', 'b', 'c', 'dot'],
        heads=[('join', 2), ('parent', 1), ('base', 1), ('norm', 1)],
        rules=[
            Rule('join_dot_l',   ('join', 'dot', X), X),
            Rule('join_dot_r',   ('join', X, 'dot'), X),
            Rule('join_root_l',  ('join', 'root', X), ('norm', X)),
            Rule('norm_norm',    ('norm', ('norm', X)), ('norm', X)),
            Rule('base_join',    ('base', ('join', X, Y)), Y),
            Rule('parent_join',  ('parent', ('join', X, Y)), X),
            Rule('join_assoc',   ('join', ('join', X, Y), Z), ('join', X, ('join', Y, Z))),
        ],
    )


_RW_PACKS = [
    _rw_pack_arith,
    _rw_pack_bool,
    _rw_pack_list,
    _rw_pack_ast,
    _rw_pack_string,
    _rw_pack_logic,
    _rw_pack_path,
]


@dataclass
class RewriteSystemConfig(Config):
    depth: int = 3
    n_insertions: int = 3
    min_rules: int = 5
    max_rules: int = 9
    min_steps: int = 2
    max_steps: int = 160
    max_chars: int = 520
    atom_keep_prob: float = 0.65
    shortcut_compression: float = 16.0

    def apply_difficulty(self, level):
        self.depth += level
        self.n_insertions += level
        self.max_rules = min(self.max_rules + level, 12)
        self.min_steps += level
        self.max_steps += 50 * level
        self.max_chars += 120 * level
        self.atom_keep_prob = max(0.35, self.atom_keep_prob - 0.05 * level)
        self.shortcut_compression = max(8.0, self.shortcut_compression - 2.0 * level)


class RewriteSystem(Task):
    summary = "Normalize term rewrite systems under boolean, list, logic, or path rules."
    def __init__(self, config=None):
        super().__init__(config=config or RewriteSystemConfig())

    def _sample_rules(self, pack, rng):
        lo = min(self.config.min_rules, len(pack.rules))
        hi = min(self.config.max_rules, len(pack.rules))
        return rng.sample(pack.rules, rng.randint(lo, hi))

    @staticmethod
    def _format_trace(terms):
        lines = [f"-> {_rw_show(u)}" for u in terms[1:]]
        lines.append(f"normal_form: {_rw_show(terms[-1])}")
        return '\n'.join(lines)

    def generate_entry(self):
        rng = random.Random()
        cfg = self.config

        for _ in range(2000):
            pack = rng.choice(_RW_PACKS)()
            rules = self._sample_rules(pack, rng)

            seed = _rw_gen_ground(cfg.depth, rng, pack.consts, pack.heads)
            nf, _ = _rw_normalize(seed, rules, cfg.max_steps)
            if nf is None:
                continue

            t = nf
            inserted = []
            for _ in range(cfg.n_insertions):
                t2, name = _rw_insert_redex(t, rules, rng, pack.consts, pack.heads, cfg.depth)
                if t2 is None:
                    continue
                if len(_rw_show(t2)) > cfg.max_chars:
                    continue
                t = t2
                inserted.append(name)

            if not inserted or t == nf:
                continue

            terms, used = _rw_trace(t, rules, cfg.max_steps)
            if terms is None or terms[-1] != nf:
                continue
            if len(used) < cfg.min_steps:
                continue

            term_s, nf_s = _rw_show(t), _rw_show(nf)
            if len(term_s) / max(1, len(nf_s)) > cfg.shortcut_compression and len(used) <= cfg.min_steps + 1:
                continue
            if not isinstance(nf, tuple) and rng.random() > cfg.atom_keep_prob:
                continue

            reverse_nf, _ = _rw_normalize(t, list(reversed(rules)), cfg.max_steps)
            rule_major_nf, _ = _rw_normalize_with(
                t, rules, _rw_step_rule_major, cfg.max_steps)
            full_nf, _ = _rw_normalize(t, pack.rules, cfg.max_steps)
            step_stats = [_rw_step_stats(u, rules) for u in terms[:-1]]
            used_counts = Counter(used)

            rules_s = _rw_rule_text(rules)
            cot = self._format_trace(terms)
            if len(rules_s) + len(term_s) + len(nf_s) + len(cot) > cfg.max_chars * 3:
                continue

            meta = edict(
                theory=pack.name,
                rules=rules_s,
                term=term_s,
                normal_form=nf_s,
                used=used,
                cot=cot,
                strategy='position_major',
                num_steps=len(used),
                num_distinct_rules_used=len(used_counts),
                used_rule_counts=dict(used_counts),
                max_rewrite_depth=max((d for d, _ in step_stats), default=0),
                overlapping_matches=sum(max(0, n - 1) for _, n in step_stats),
                answer_is_input_subterm=_rw_is_subterm(nf, t, proper=True),
                reverse_order_same_nf=reverse_nf == nf,
                rule_major_same_nf=rule_major_nf == nf,
                full_pack_same_nf=full_nf == nf,
                omitted_rule_count=len(pack.rules) - len(rules),
                split_key=_rw_split_key(rules, t, nf),
            )
            meta.payload = {"rules": rules_s, "term": term_s}
            return Entry(metadata=meta, answer=nf_s)

        raise RuntimeError("could not sample a rewrite-system instance")

    def render_prompt(self, metadata):
        return (
            "Normalize by the ordered rewrite rules. At each step, scan subterm "
            "positions outermost-first and left-to-right. Stop at the first position "
            "matched by at least one rule, then apply the earliest matching rule in "
            "the listed order (position priority first; rule priority second).\n\n"
            f"{render_payload(metadata['payload'])}\n\n"
            "The answer is the normal form."
        )

    def deduplication_key(self, problem):
        return problem.metadata["split_key"]

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0

        ans = str(answer).strip()
        ref = str(entry.answer).strip()

        if ans == ref:
            return 1.0

        if 'normal_form:' in ans:
            ans = ans.rsplit('normal_form:', 1)[-1].strip()

        try:
            got = _rw_parse(ans)
            target = _rw_parse(entry.metadata['normal_form'])
            return float(got == target)
        except Exception:
            return float(ans.replace(' ', '') == entry.metadata['normal_form'].replace(' ', ''))


# ─────────────────────────────────────────────────────────────────────────────
# First-order MGU implication
# ─────────────────────────────────────────────────────────────────────────────
# term ::= variable "xN" | constant "a" | function tuple ("f", arg, ...)

_MGU_ARITIES = {'f': 1, 'g': 1, 'h': 2, 'p': 2, 'q': 3, 'r': 3}
_MGU_CONSTS = ('a', 'b', 'c')


def _mgu_is_var(t):
    return isinstance(t, str) and len(t) > 1 and t[0] == 'x' and t[1:].isdigit()


def _mgu_show(t):
    if isinstance(t, tuple):
        return f"{t[0]}({', '.join(_mgu_show(x) for x in t[1:])})"
    return str(t)


def _mgu_vars(t):
    if _mgu_is_var(t):
        return {t}
    if isinstance(t, tuple):
        return set().union(*(_mgu_vars(x) for x in t[1:])) if len(t) > 1 else set()
    return set()


def _mgu_symbols(t):
    funcs, consts = set(), set()
    if isinstance(t, tuple):
        funcs.add(t[0])
        for x in t[1:]:
            f, c = _mgu_symbols(x)
            funcs |= f
            consts |= c
    elif not _mgu_is_var(t):
        consts.add(t)
    return funcs, consts


def _mgu_depth(t):
    if not isinstance(t, tuple):
        return 0
    return 1 + max((_mgu_depth(x) for x in t[1:]), default=0)


def _mgu_positions(t, path=()):
    yield path, t
    if isinstance(t, tuple):
        for i, x in enumerate(t[1:], 1):
            yield from _mgu_positions(x, path + (i,))


def _mgu_replace(t, path, new):
    if not path:
        return new
    args = list(t[1:])
    args[path[0] - 1] = _mgu_replace(args[path[0] - 1], path[1:], new)
    return (t[0], *args)


def _mgu_apply(subst, t):
    if _mgu_is_var(t):
        seen = set()
        while _mgu_is_var(t) and t in subst and t not in seen:
            seen.add(t)
            t = subst[t]
        return _mgu_apply(subst, t) if isinstance(t, tuple) else t
    if isinstance(t, tuple):
        return (t[0], *[_mgu_apply(subst, x) for x in t[1:]])
    return t


def _mgu_occurs(v, t, subst):
    t = _mgu_apply(subst, t)
    if t == v:
        return True
    return isinstance(t, tuple) and any(_mgu_occurs(v, x, subst) for x in t[1:])


def _mgu_bind(subst, v, t, stats):
    t = _mgu_apply(subst, t)
    if t == v:
        return subst
    if _mgu_occurs(v, t, subst):
        raise ValueError("occurs check failed")
    subst = {k: _mgu_apply({v: t}, val) for k, val in subst.items()}
    subst[v] = t
    stats['bindings'] += 1
    if _mgu_is_var(t):
        stats['alias_bindings'] += 1
    return subst


def _mgu_unify(equations):
    subst, stats = {}, Counter()
    stack = list(equations)
    while stack:
        a, b = stack.pop()
        a, b = _mgu_apply(subst, a), _mgu_apply(subst, b)
        if a == b:
            continue
        if _mgu_is_var(a):
            subst = _mgu_bind(subst, a, b, stats)
        elif _mgu_is_var(b):
            subst = _mgu_bind(subst, b, a, stats)
        elif (isinstance(a, tuple) and isinstance(b, tuple) and
              a[0] == b[0] and len(a) == len(b)):
            stats['decompositions'] += 1
            stack.extend(zip(a[1:], b[1:]))
        else:
            raise ValueError("not unifiable")
    return {k: _mgu_apply(subst, v) for k, v in subst.items()}, stats


def _mgu_candidate_answer(equations, candidate):
    sol, stats = _mgu_unify(equations)
    left = _mgu_apply(sol, candidate[0])
    right = _mgu_apply(sol, candidate[1])
    return ("yes" if left == right else "no"), sol, stats


def _mgu_sample_ground(depth, rng, force_func=False):
    if depth <= 0 or (not force_func and rng.random() < 0.30):
        return rng.choice(_MGU_CONSTS)
    funcs = [('f', 1), ('g', 1), ('h', 2), ('p', 2)]
    if depth >= 2:
        funcs += [('q', 3), ('r', 3)]
    f, arity = rng.choice(funcs)
    return (f, *[_mgu_sample_ground(depth - 1, rng) for _ in range(arity)])


def _mgu_new_var(state):
    v = f"x{state['next_var']}"
    state['next_var'] += 1
    return v


def _mgu_mask(t, rng, state, mask_prob, share_prob, root=False):
    if not root and rng.random() < mask_prob:
        old = state['by_term'].get(t, [])
        if old and rng.random() < share_prob:
            return rng.choice(old)
        v = _mgu_new_var(state)
        state['by_term'].setdefault(t, []).append(v)
        return v
    if isinstance(t, tuple):
        return (t[0], *[_mgu_mask(x, rng, state, mask_prob, share_prob) for x in t[1:]])
    return t


def _mgu_common_instance_equation(cfg, rng, state):
    for _ in range(80):
        local = {'next_var': state['next_var'], 'by_term': {}}
        root = rng.choice(['p', 'h'] if cfg.decomposition_depth <= 2 else ['p', 'h', 'q', 'r'])
        latent = (root, *[_mgu_sample_ground(max(0, cfg.rhs_depth - 1), rng)
                         for _ in range(_MGU_ARITIES[root])])
        share = min(0.85, 0.20 + 0.12 * cfg.sharing_count)
        left = _mgu_mask(latent, rng, local, 0.58, share, root=True)
        right = _mgu_mask(latent, rng, local, 0.62, share, root=True)
        if _mgu_vars(left) and _mgu_vars(right) and left != right:
            state['next_var'] = local['next_var']
            return left, right
    raise RuntimeError("could not mask common instance")


def _mgu_equation_vars(eq):
    return _mgu_vars(eq[0]) | _mgu_vars(eq[1])


def _mgu_connected_instance_equation(cfg, rng, state, equations, core_vars):
    _, sol, _ = _mgu_candidate_answer(equations, ('x0', 'x0'))
    solved = [(v, _mgu_apply(sol, v)) for v in sorted(core_vars)
              if v in sol and _mgu_apply(sol, v) != v]
    if not solved:
        raise RuntimeError("no connected variable")

    for _ in range(80):
        v, t = rng.choice(solved)
        local = {'next_var': state['next_var'], 'by_term': {}}
        root = rng.choice(['p', 'h'] if cfg.decomposition_depth <= 2 else ['p', 'h', 'q', 'r'])
        arity = _MGU_ARITIES[root]
        pos = rng.randrange(arity)
        args = [_mgu_sample_ground(max(0, cfg.rhs_depth - 1), rng) for _ in range(arity)]
        args[pos] = t
        latent = (root, *args)
        share = min(0.85, 0.20 + 0.12 * cfg.sharing_count)
        left = list(_mgu_mask(latent, rng, local, 0.58, share, root=True)[1:])
        right = list(_mgu_mask(latent, rng, local, 0.62, share, root=True)[1:])
        if rng.random() < 0.5:
            left[pos] = v
        else:
            right[pos] = v
        eq = (root, *left), (root, *right)
        if _mgu_equation_vars(eq) - core_vars:
            state['next_var'] = local['next_var']
            return eq
    raise RuntimeError("could not build connected equation")


def _mgu_context_pair(a, b, rng, funcs, consts):
    funcs = list(funcs) or list(_MGU_ARITIES)
    consts = list(consts) or list(_MGU_CONSTS)
    f = rng.choice(funcs)
    arity = _MGU_ARITIES[f]
    if arity == 1:
        return (f, a), (f, b)
    args = [rng.choice(consts) for _ in range(arity)]
    pos = rng.randrange(arity)
    left, right = list(args), list(args)
    left[pos], right[pos] = a, b
    return (f, *left), (f, *right)


def _mgu_mutate(t, rng, funcs, consts, distance=1):
    out = t
    for _ in range(max(1, distance)):
        positions = list(_mgu_positions(out))
        rng.shuffle(positions)
        changed = False
        for path, sub in positions:
            if not isinstance(sub, tuple) and not _mgu_is_var(sub):
                choices = [c for c in consts if c != sub]
                if choices:
                    out = _mgu_replace(out, path, rng.choice(choices))
                    changed = True
                    break
            if isinstance(sub, tuple):
                choices = [f for f in funcs if f != sub[0] and _MGU_ARITIES[f] == len(sub) - 1]
                if choices:
                    out = _mgu_replace(out, path, (rng.choice(choices), *sub[1:]))
                    changed = True
                    break
        if not changed:
            unary = [f for f in funcs if _MGU_ARITIES[f] == 1]
            if unary:
                path, sub = rng.choice(positions)
                out = _mgu_replace(out, path, (rng.choice(unary), sub))
            else:
                return None
    return out


def _mgu_mutate_close(t, rng, funcs, consts):
    funcs = set(funcs)
    consts = set(consts)
    positions = list(_mgu_positions(t))
    leaf_positions = [(p, x) for p, x in positions
                      if p and not isinstance(x, tuple) and not _mgu_is_var(x)]
    rng.shuffle(leaf_positions)
    for path, sub in leaf_positions:
        choices = [c for c in consts if c != sub]
        if choices:
            return _mgu_replace(t, path, rng.choice(choices))

    inner_funcs = [(p, x) for p, x in positions if p and isinstance(x, tuple)]
    rng.shuffle(inner_funcs)
    for path, sub in inner_funcs:
        choices = [f for f in funcs if f != sub[0] and _MGU_ARITIES[f] == len(sub) - 1]
        if choices:
            return _mgu_replace(t, path, (rng.choice(choices), *sub[1:]))
    return None


def _mgu_one_step_pairs(equations):
    out = set(equations) | {(b, a) for a, b in equations}
    for a, b in equations:
        if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
            out.update(zip(a[1:], b[1:]))
            out.update((y, x) for x, y in zip(a[1:], b[1:]))
    return out


def _mgu_surface_decides(candidate):
    a, b = candidate
    if a == b:
        return True
    if not _mgu_vars(a) and not _mgu_vars(b):
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        return a[0] != b[0] or len(a) != len(b)
    return False


def _mgu_all_symbols(equations):
    funcs, consts = set(), set()
    for a, b in equations:
        for t in (a, b):
            f, c = _mgu_symbols(t)
            funcs |= f
            consts |= c
    return funcs, consts


def _mgu_rename_term(t, vmap, fmap, cmap):
    if _mgu_is_var(t):
        return vmap.get(t, t)
    if isinstance(t, tuple):
        return (fmap.get(t[0], t[0]), *[_mgu_rename_term(x, vmap, fmap, cmap) for x in t[1:]])
    return cmap.get(t, t)


def _mgu_rename(equations, candidate, rng):
    vars_ = sorted(set().union(*(_mgu_vars(t) for eq in equations for t in eq), *(_mgu_vars(t) for t in candidate)))
    new_vars = [f"x{i}" for i in range(len(vars_))]
    rng.shuffle(new_vars)
    vmap = dict(zip(vars_, new_vars))

    funcs, consts = _mgu_all_symbols(equations + [candidate])
    cmap_vals = list(_MGU_CONSTS)
    rng.shuffle(cmap_vals)
    cmap = dict(zip(sorted(consts), cmap_vals))
    fmap = {}
    for arity in {1, 2, 3}:
        old = sorted(f for f in funcs if _MGU_ARITIES[f] == arity)
        new = [f for f, a in _MGU_ARITIES.items() if a == arity]
        rng.shuffle(new)
        fmap.update(dict(zip(old, new)))

    req = [(_mgu_rename_term(a, vmap, fmap, cmap), _mgu_rename_term(b, vmap, fmap, cmap))
           for a, b in equations]
    rcand = (_mgu_rename_term(candidate[0], vmap, fmap, cmap),
             _mgu_rename_term(candidate[1], vmap, fmap, cmap))
    return req, rcand


def _mgu_tree_diff(a, b):
    if a == b:
        return 0
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
        return sum(_mgu_tree_diff(x, y) for x, y in zip(a[1:], b[1:]))
    return 1


def _mgu_candidate_features(candidate):
    a, b = candidate
    av, bv = _mgu_vars(a), _mgu_vars(b)
    return {
        'left_depth': _mgu_depth(a),
        'right_depth': _mgu_depth(b),
        'left_variable_count': len(av),
        'right_variable_count': len(bv),
        'left_ground': not av,
        'right_ground': not bv,
        'same_root_symbol': (isinstance(a, tuple) and isinstance(b, tuple)
                             and a[0] == b[0]),
        'shared_variable_count': len(av & bv),
        'candidate_tree_diff': _mgu_tree_diff(a, b),
        'candidate_length': len(_mgu_show(a)) + len(_mgu_show(b)),
    }


def _mgu_split_key(equations, candidate, answer):
    """Exact canonical form under symbol renaming, equation order, and equality sides."""
    def copy_maps(maps):
        variables, constants, functions = maps
        return variables.copy(), constants.copy(), {arity: values.copy() for arity, values in functions.items()}

    def canon_term(t, maps):
        variables, constants, functions = maps
        if _mgu_is_var(t):
            variables.setdefault(t, len(variables))
            return "V", variables[t]
        if isinstance(t, tuple):
            arity = len(t) - 1
            names = functions.setdefault(arity, {})
            names.setdefault(t[0], len(names))
            return ("F", arity, names[t[0]], *(canon_term(x, maps) for x in t[1:]))
        constants.setdefault(t, len(constants))
        return "C", constants[t]

    def pair_options(pair, maps):
        out = []
        for left, right in (pair, pair[::-1]):
            next_maps = copy_maps(maps)
            out.append(((canon_term(left, next_maps), canon_term(right, next_maps)), next_maps))
        best = min(value for value, _ in out)
        return [(value, state) for value, state in out if value == best]

    def canon_equations(remaining, maps):
        if not remaining:
            return ()
        choices = []
        for i, equation in enumerate(remaining):
            rest = remaining[:i] + remaining[i + 1:]
            for value, state in pair_options(equation, maps):
                choices.append((value, rest, state))
        first = min(value for value, _, _ in choices)
        return min((value, *canon_equations(rest, state))
                   for value, rest, state in choices if value == first)

    empty = ({}, {}, {})
    candidates = []
    for candidate_value, maps in pair_options(candidate, empty):
        candidates.append((candidate_value, canon_equations(tuple(equations), maps), answer))
    canonical = min(candidates)
    return hashlib.sha1(repr(canonical).encode()).hexdigest()


def _mgu_close_negative(sol, candidate):
    a = _mgu_apply(sol, candidate[0])
    b = _mgu_apply(sol, candidate[1])
    if a == b:
        return False
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
        return abs(_mgu_depth(a) - _mgu_depth(b)) <= 1 and _mgu_tree_diff(a, b) <= 2
    return False


def _mgu_positive_candidate(cfg, rng, equations, sol, allowed_vars, allow_same=True):
    funcs, consts = _mgu_all_symbols(equations)
    solved = [(v, _mgu_apply(sol, v)) for v in sorted(sol)
              if v in allowed_vars and _mgu_apply(sol, v) != v]
    groups = {}
    for v, t in solved:
        groups.setdefault(repr(t), []).append((v, t))
    same_pairs = [(xs[0][0], xs[1][0], xs[0][1]) for xs in groups.values() if len(xs) >= 2]
    if not solved:
        raise RuntimeError("no solved variables")

    for _ in range(80):
        kinds = ['image', 'wrapped', 'masked']
        if allow_same:
            kinds.append('same_var')
        kind = rng.choice(kinds)
        if kind == 'same_var' and same_pairs:
            v, w, _ = rng.choice(same_pairs)
            cand = (v, w)
        else:
            v, t = rng.choice(solved)
            if kind in ('wrapped', 'masked'):
                other = rng.choice([w for w, u in solved if u == t] or [t])
                cand = _mgu_context_pair(v, other, rng, funcs, consts)
            else:
                cand = (v, t)
        if not _mgu_surface_decides(cand):
            return cand
    raise RuntimeError("could not build positive candidate")


def _mgu_candidate(cfg, rng, equations, sol, want_positive, allowed_vars):
    funcs, consts = _mgu_all_symbols(equations)
    solved = [(v, _mgu_apply(sol, v)) for v in sorted(sol)
              if v in allowed_vars and _mgu_apply(sol, v) != v]
    for _ in range(80):
        if want_positive:
            return _mgu_positive_candidate(cfg, rng, equations, sol, allowed_vars)

        pos = _mgu_positive_candidate(cfg, rng, equations, sol, allowed_vars, allow_same=False)
        bad = _mgu_mutate_close(_mgu_apply(sol, pos[1]), rng, funcs, consts)
        if bad is not None:
            cand = (pos[0], bad)
            if not _mgu_surface_decides(cand) and (cfg.level < 2 or _mgu_close_negative(sol, cand)):
                return cand

        if cfg.level < 2 and solved:
            v, t = rng.choice(solved)
            bad = _mgu_mutate(t, rng, funcs, consts, cfg.candidate_distance)
            if bad is not None:
                cand = (v, bad)
                if not _mgu_surface_decides(cand):
                    return cand
    raise RuntimeError("could not build candidate")


def _mgu_core_size(equations, candidate, answer):
    core = 0
    for i in range(len(equations)):
        try:
            ans, _, _ = _mgu_candidate_answer(equations[:i] + equations[i + 1:], candidate)
        except ValueError:
            core += 1
            continue
        core += int(ans != answer)
    return core


def _mgu_validate(cfg, equations, candidate, answer, sol, stats):
    funcs, consts = _mgu_all_symbols(equations)
    cfuncs, cconsts = _mgu_all_symbols([candidate])
    if not cfuncs <= funcs or not cconsts <= consts:
        return False
    if not (_mgu_vars(candidate[0]) | _mgu_vars(candidate[1])) <= set().union(*(_mgu_vars(t) for eq in equations for t in eq)):
        return False
    if _mgu_surface_decides(candidate):
        return False
    if cfg.level >= 2 and candidate in _mgu_one_step_pairs(equations):
        return False
    meaningful = sum(_mgu_apply(sol, v) != v for v in sol)
    if cfg.level >= 2 and meaningful < 2:
        return False
    if stats['bindings'] + stats['decompositions'] < cfg.min_trace_steps:
        return False
    if answer == 'yes' and cfg.level >= 2 and _mgu_core_size(equations, candidate, answer) == 0:
        return False
    if answer == 'no' and cfg.level >= 2 and not _mgu_close_negative(sol, candidate):
        return False
    return True


@dataclass
class UnificationEntailmentConfig(Config):
    n_bindings: int = 1
    rhs_depth: int = 1
    alias_chain_len: int = 0
    sharing_count: int = 0
    decomposition_depth: int = 1
    num_redundant_equations: int = 0
    num_distractor_equations: int = 0
    candidate_distance: int = 1
    min_trace_steps: int = 2
    max_equations: int = 6

    def apply_difficulty(self, level):
        self.n_bindings += level
        self.rhs_depth += level // 2
        self.alias_chain_len += level // 2
        self.sharing_count += level
        self.decomposition_depth += level
        self.num_redundant_equations += (level + 1) // 2
        self.num_distractor_equations += level // 3
        self.candidate_distance += level // 3
        self.min_trace_steps += level
        self.max_equations += level


class UnificationEntailment(Task):
    summary = "Decide if an equality is implied by the most general unifier of equations."
    config_cls = UnificationEntailmentConfig

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config=config or UnificationEntailmentConfig(), *args, **kwargs)

    def _build_equations(self, rng):
        cfg = self.config
        state = {'next_var': 0, 'by_term': {}}
        equations, tags, core_vars = [], [], set()

        def add(eq, tag):
            if len(equations) < cfg.max_equations:
                equations.append(eq)
                tags.append(tag)
                if tag != 'distractor':
                    core_vars.update(_mgu_equation_vars(eq))

        if cfg.level <= 1 and rng.random() < 0.35:
            v = _mgu_new_var(state)
            add((('f', v), ('f', rng.choice(_MGU_CONSTS))), 'core')
        else:
            add(_mgu_common_instance_equation(cfg, rng, state), 'core')
            for _ in range(max(0, min(cfg.n_bindings, 2 + cfg.level // 2) - 1)):
                add(_mgu_connected_instance_equation(cfg, rng, state, equations, core_vars), 'core')

        for _ in range(cfg.alias_chain_len):
            if not core_vars:
                break
            v = rng.choice(sorted(core_vars))
            w = _mgu_new_var(state)
            add((w, v), 'alias')

        for _ in range(min(cfg.sharing_count, 2)):
            ans, sol, _ = _mgu_candidate_answer(equations, ('x0', 'x0'))
            solved = [(v, _mgu_apply(sol, v)) for v in sol
                      if v in core_vars and _mgu_apply(sol, v) != v]
            if not solved:
                continue
            v, t = rng.choice(solved)
            y = _mgu_new_var(state)
            add((('p', v, v), ('p', t, y)), 'sharing')

        for _ in range(cfg.num_redundant_equations):
            _, sol, _ = _mgu_candidate_answer(equations, ('x0', 'x0'))
            solved = [(v, _mgu_apply(sol, v)) for v in sol
                      if v in core_vars and _mgu_apply(sol, v) != v]
            if not solved:
                continue
            v, t = rng.choice(solved)
            funcs, consts = _mgu_all_symbols(equations)
            add(_mgu_context_pair(v, t, rng, funcs, consts), 'redundant')

        for _ in range(cfg.num_distractor_equations):
            add(_mgu_common_instance_equation(cfg, rng, state), 'distractor')

        return equations, Counter(tags), set(core_vars)

    def generate_entry(self):
        rng = random.Random()
        cfg = self.config
        target = 'yes' if rng.random() < 0.5 else 'no'
        for _ in range(6000):
            try:
                equations, eq_tags, core_vars = self._build_equations(rng)
                _, sol, stats = _mgu_candidate_answer(equations, ('x0', 'x0'))
                candidate = _mgu_candidate(cfg, rng, equations, sol, target == 'yes', core_vars)
                equations = [(b, a) if rng.random() < 0.5 else (a, b)
                             for a, b in equations]
                if rng.random() < 0.5:
                    candidate = candidate[1], candidate[0]
                equations, candidate = _mgu_rename(equations, candidate, rng)
                answer, sol, stats = _mgu_candidate_answer(equations, candidate)
            except (RuntimeError, ValueError):
                continue
            if answer != target:
                continue
            if not _mgu_validate(cfg, equations, candidate, answer, sol, stats):
                continue

            eq_s = [f"{_mgu_show(a)} = {_mgu_show(b)}" for a, b in equations]
            cand_s = f"{_mgu_show(candidate[0])} = {_mgu_show(candidate[1])}"
            all_terms = [t for eq in equations for t in eq] + [candidate[0], candidate[1]]
            funcs, consts = _mgu_all_symbols(equations)
            core = _mgu_core_size(equations, candidate, answer) if answer == 'yes' else 0
            meaningful = sum(_mgu_apply(sol, v) != v for v in sol)
            candidate_features = _mgu_candidate_features(candidate)
            canonical_answer = answer.capitalize()
            meta = edict(
                answer=canonical_answer,
                equations=eq_s,
                candidate=cand_s,
                num_equations=len(equations),
                num_variables=len(set().union(*(_mgu_vars(t) for t in all_terms))),
                num_bindings_in_mgu=meaningful,
                max_term_depth=max(_mgu_depth(t) for t in all_terms),
                candidate_depth=max(_mgu_depth(candidate[0]), _mgu_depth(candidate[1])),
                trace_steps=stats['bindings'] + stats['decompositions'],
                num_decompositions=stats['decompositions'],
                num_alias_bindings=stats['alias_bindings'],
                num_sharing_constraints=eq_tags['sharing'],
                num_redundant_equations=eq_tags['redundant'],
                difficulty_level=cfg.level,
                positive_or_negative='positive' if answer == 'yes' else 'negative',
                generation_mode='common_instance_masking',
                minimal_core_size_estimate=core,
                num_constants=len(consts),
                num_function_symbols=len(funcs),
                candidate_features=candidate_features,
                split_key=_mgu_split_key(equations, candidate, answer),
            )
            return Entry(metadata=meta, answer=canonical_answer)
        raise RuntimeError("could not sample an MGU implied-equality instance")

    def render_prompt(self, metadata):
        equations = '\n'.join(f"- {e}" for e in metadata['equations'])
        return (
            "Compute a most general unifier of the equations. Apply it to both sides "
            "of the candidate equality. Answer Yes if the instantiated candidate "
            "terms are identical, otherwise answer No. The equations are guaranteed "
            "to be unifiable.\n\n"
            "Equations:\n"
            f"{equations}\n\n"
            "Candidate:\n"
            f"{metadata['candidate']}"
        )

    def deduplication_key(self, problem):
        return problem.metadata["split_key"]

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        ans = str(answer).strip().lower().rstrip('.')
        return float(ans == entry.answer.lower())
