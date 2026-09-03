"""Term unification: ask what a named variable is bound to by the most general
unifier of two first-order terms, or whether they unify at all."""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

FUNCS = ["f", "g", "h"]
CONSTS = ["a", "b", "c"]
VARS = ["x", "y", "z", "u", "v", "w"]


def render_term(t):
    if t[0] == "var":
        return t[1]
    if t[0] == "const":
        return t[1]
    _, name, args = t
    return "%s(%s)" % (name, ",".join(render_term(a) for a in args))


class NoUnify(Exception):
    pass


def unify(term1, term2):
    """Standard Robinson unification (with occurs check). Returns dict
    var-name -> term-in-internal-repr, or None if the terms do not unify."""
    subst = {}

    def walk(t):
        while t[0] == "var" and t[1] in subst:
            t = subst[t[1]]
        return t

    def occurs(x, t):
        if t[0] == "var":
            return t[1] == x
        if t[0] == "const":
            return False
        return any(occurs(x, a) for a in t[2])

    def apply_subst(t, m):
        t = walk(t)
        if t[0] == "var":
            if t[1] in m:
                return walk(m[t[1]])
            return t
        if t[0] == "const":
            return t
        return (t[0], t[1], [apply_subst(a, m) for a in t[2]])

    def put(vname, t):
        t = walk(t)
        if t[0] == "var" and t[1] == vname:
            return
        if occurs(vname, t):
            raise NoUnify
        subst[vname] = t
        for k in [kk for kk in subst if kk != vname]:
            subst[k] = apply_subst(subst[k], {vname: t})

    def run(t1, t2):
        t1 = walk(t1)
        t2 = walk(t2)
        if t1[0] == "var" and t2[0] == "var" and t1[1] == t2[1]:
            return
        if t1[0] == "var":
            put(t1[1], t2)
            return
        if t2[0] == "var":
            put(t2[1], t1)
            return
        if t1[0] != t2[0]:
            raise NoUnify
        if t1[0] == "const":
            if t1[1] != t2[1]:
                raise NoUnify
            return
        if t1[1] != t2[1] or len(t1[2]) != len(t2[2]):
            raise NoUnify
        for a1, a2 in zip(t1[2], t2[2]):
            run(a1, a2)

    try:
        run(term1, term2)
    except NoUnify:
        return None
    return subst


def _get_path(t, path):
    for i in path:
        t = t[2][i]
    return t


def _is_func_only(t):
    return t[0] == "func"


def _replace_path(t, path, node):
    if not path:
        return node
    return (t[0], t[1], [_replace_path(a, path[1:], node) if j == path[0] else a
                          for j, a in enumerate(t[2])])


def _contains_var(t, v):
    if t[0] == "var":
        return t[1] == v
    if t[0] == "const":
        return False
    return any(_contains_var(a, v) for a in t[2])


def _vars_of(t):
    if t[0] == "var":
        yield t[1]
        return
    if t[0] == "const":
        return
    for a in t[2]:
        for v in _vars_of(a):
            yield v


def _term_vars(terms):
    s = set()
    for t in terms:
        for v in _vars_of(t):
            s.add(v)
    return s


def _nodes(t, path=()):
    """Yield all (path, term) node positions of t, root first."""
    yield (path, t)
    if t[0] == "func":
        for i, a in enumerate(t[2]):
            for p, node in _nodes(a, tuple(path) + (i,)):
                yield (p, node)


def _random_term(max_depth, n_vars, rng):
    """Build a term with up to max_depth function layers, but always containing
    at least one function node when max_depth >= 1, on a deterministic choice."""
    if max_depth == 0:
        return ("const", rng.choice(CONSTS))
    if max_depth == 1:
        node = ("func", rng.choice(FUNCS), [])
        node = (node[0], node[1], [
            rng.choice([("var", rng.choice(VARS[:n_vars])), ("const", rng.choice(CONSTS))])
            for _ in range(rng.randint(1, 2))
        ])
        return node
    node = ("func", rng.choice(FUNCS), [])
    args = []
    for _ in range(rng.randint(1, 2)):
        if rng.random() < 0.45:
            args.append(("var", rng.choice(VARS[:n_vars])))
        else:
            args.append(_random_term(max_depth - 1, n_vars, rng))
    return (node[0], node[1], args)


@dataclass
class TermUnificationConfig(Config):
    max_depth: int = 2
    n_vars: int = 3

    def apply_difficulty(self, level):
        self.max_depth = sround(int(self.max_depth) + level)
        self.n_vars = sround(int(self.n_vars) + level)


class TermUnification(Task):
    config_cls = TermUnificationConfig

    def generate_entry(self):
        cfg = self.config
        depth = int(cfg.max_depth)
        n_vars = int(cfg.n_vars)

        if random.random() < 0.2:
            fail_type = random.choice(["arity", "symbol", "occurs"])
            return self._build_failing(fail_type, depth, n_vars)
        return self._build_unifiable(depth, n_vars)

    def _build_unifiable(self, depth, n_vars):
        # Guarantee unification: t1 is any term, t2 is t1 with one internal
        # function subtree replaced by the target variable. The most general
        # unifier then binds target to that subtree (a compound term), keeping
        # answers structurally varied.
        target = random.choice(VARS[:n_vars])
        for _ in range(100):
            t1 = _random_term(depth, n_vars, random)
            candidates = [p for p, node in _nodes(t1)
                          if _is_func_only(node) and not _contains_var(node, target)]
            if not candidates:
                continue
            path = random.choice(candidates)
            subtree = _get_path(t1, path)
            t2 = _replace_path(t1, path, ("var", target))
            subst = unify(t1, t2)
            if subst is not None and target in subst:
                binding = subst[target]
                if not _contains_var(binding, target):
                    return self._make_entry(t1, t2, target, binding)
        raise RuntimeError("term_unification: could not build unifiable example")

    def _build_failing(self, fail_type, depth, n_vars):
        target = random.choice(VARS[:n_vars])
        for _ in range(100):
            if fail_type == "arity":
                t1 = ("func", "f", [("var", target)])
                t2 = ("func", "f", [("var", target), ("var", "u")])
            elif fail_type == "symbol":
                t1 = ("func", "f", [("var", target)])
                t2 = ("func", "g", [("var", target)])
            else:  # occurs
                t1 = ("var", target)
                t2 = ("func", "f", [("var", target)])
            if unify(t1, t2) is not None:
                continue
            return self._make_fail_entry(t1, t2, target)
        raise RuntimeError("term_unification: could not build failing example")

    @staticmethod
    def _make_entry(t1, t2, target, binding):
        bstr = render_term(binding)
        metadata = edict({
            "payload": {
                "t1": render_term(t1),
                "t2": render_term(t2),
                "target": target,
                "vars": sorted(_term_vars([t1, t2])),
            },
            "binding": bstr,
        })
        return Entry(metadata=metadata, answer=bstr)

    @staticmethod
    def _make_fail_entry(t1, t2, target):
        metadata = edict({
            "payload": {
                "t1": render_term(t1),
                "t2": render_term(t2),
                "target": target,
                "vars": sorted(_term_vars([t1, t2])),
            },
            "binding": "none",
        })
        return Entry(metadata=metadata, answer="none")

    def render_prompt(self, metadata):
        p = metadata.payload
        return (
            "Terms are built from variables %s, constants %s, and functions %s.\n"
            "Determine whether the two terms below unify. If they do, give the "
            "most general unifier's binding for the variable %s, written in the "
            "same syntax (e.g. f(x,b)).\n\n"
            "Term 1: %s\nTerm 2: %s\n\n"
            "If they do not unify, answer with exactly \"none\". Otherwise "
            "answer with the term that %s is bound to."
            % (
                ", ".join(p["vars"]),
                ", ".join(CONSTS),
                ", ".join(FUNCS),
                p["target"],
                p["t1"],
                p["t2"],
                p["target"],
            )
        )

    def score_answer(self, answer, entry):
        true = entry.metadata["binding"]
        if true == "none":
            return 1.0 if answer.strip().lower() == "none" else 0.0
        answer = answer.strip().replace(" ", "")
        true = true.replace(" ", "")
        return 1.0 if answer == true else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Add first-order unification over small symbolic terms.',
 'hypothesis': 'S27',
 'changes': 'Ask what a named variable is bound to by the most general unifier '
            'of two terms, or whether they unify at all.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1680339059,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
