"""Metamath set.mm proof-tree tasks.

The generator downloads set.mm into appdirs, mines small assertions as inference
rules, builds fresh uncompressed proof trees by substitution, and checks them
with a small Metamath stack verifier.
"""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve

from appdirs import AppDirs
from easydict import EasyDict as edict

from reasoning_core.template import Config, Entry, Task, render_payload


_SET_MM_URL = "https://raw.githubusercontent.com/metamath/set.mm/develop/set.mm"
_TYPECODES = {"|-", "wff", "class", "set"}
_VARS = {
    "wff": ("ph", "ps", "ch", "th", "ta", "et", "ze", "si"),
    "class": ("A", "B", "C", "D", "E", "F", "G", "H"),
    "set": ("x", "y", "z", "w", "v", "u", "t", "s"),
}
_MATH_TOKENS = {
    "=", "e.", "C_", "<", "<_", ">", ">_", "+", "-", "x.", "/", "0", "1", "2",
    "RR", "CC", "NN", "ZZ", "QQ", "dom", "ran", "Fun", "Rel", "Ord", "On", "U.",
    "|^|", "i^i", "X.", "sSet", "Top", "Grp", "Mgm", "Ring", "Field",
    "Abel", "CMnd", "Mnd", "sin", "cos", "tan", "exp", "log", "`",
}
_PURE_WFF = {"ph", "ps", "ch", "th", "ta", "et", "ze", "si"}
_KEEP_HEADS = {"e.", "=", "=/=", "<", "<_", ">", ">_", "C_"}
_KEEP_CONSTS = {"RR", "CC", "NN", "ZZ", "QQ", "Abel", "CMnd", "Mnd", "Grp", "Ring", "Field"}
_REL_TOKENS = {"=", "=/=", "e.", "C_", "C.", "<", "<_", ">", ">_"}
_LOGIC_PRED_TOKENS = {"<->", "/\\", "\\/"}
_BIN_FUNC_TOKENS = {"+", "-", "x.", "/", "^"}
_OBJECT_CONSTS = {"0", "1", "2", "+oo", "-oo"}
_PREFIX_APP = {
    "*": "conjugate",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "-u": "-",
}
_PRECEDENCE = [
    ("->", "<->"),
    ("/\\", "\\/"),
    ("=", "=/=", "e.", "C_", "C.", "<", "<_", ">", ">_"),
    ("+", "-"),
    ("x.", "/"),
    ("^",),
]
_DISPLAY_VARS = {
    "ph": "ctx", "ps": "hyp", "ch": "p", "th": "q", "ta": "r", "et": "h1", "ze": "h2", "si": "h3",
}
_CONTEXT_NAMES = ("ctx", "hyp", "p", "q", "r", "h1", "h2", "h3")
_OBJECT_NAMES = ("x", "y", "z", "u", "v", "s", "t", "x1", "y1", "z1")
_MAX_CLOSURE_FORMULAS = 500
_MAX_CLOSURE_MATCHES = 50_000
_DV_RELAX_STATS = Counter()


@dataclass(frozen=True)
class MMRule:
    label: str
    conclusion: tuple[str, ...]
    floating: tuple[tuple[str, str, str], ...]  # label, typecode, variable
    essential: tuple[tuple[str, ...], ...]
    dv: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class MMTree:
    target: tuple[str, ...]
    leaves: tuple[tuple[str, ...], ...]
    proof: tuple[str, ...]
    used: frozenset[str]


@dataclass
class MetamathConfig(Config):
    proof_depth: int = 2
    n_rules: int = 5
    n_options: int = 4
    max_premises: int = 6
    formula_len_cap: int = 18

    def apply_difficulty(self, level):
        self.proof_depth += level
        self.n_rules += level
        self.max_premises += level
        self.formula_len_cap += 2 * level


def _metamath_dir():
    return Path(AppDirs("reasoning_core").user_data_dir) / "metamath"


def ensure_set_mm():
    """Download set.mm once into appdirs and return its path."""
    base = _metamath_dir()
    path = base / "set.mm"
    if not path.exists() or path.stat().st_size < 1_000_000:
        base.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urlretrieve(_SET_MM_URL, tmp)
        tmp.replace(path)
    return path


def _tokens(text):
    text = re.sub(r"\$\((?:.|\n)*?\$\)", " ", text)
    return re.findall(r"\S+", text)


def _pairs(xs):
    return {tuple(sorted(p)) for p in itertools.combinations(xs, 2)}


def _vars_in(expr, active_vars):
    return {t for t in expr if t in active_vars}


@lru_cache(maxsize=1)
def _database():
    toks = _tokens(ensure_set_mm().read_text(encoding="utf-8", errors="ignore"))
    frames = [edict(vars=set(), f={}, f_order=[], e=[], dv=set())]
    labels, rules, var_type, var_flabel = {}, {}, {}, {}
    constants, i = set(), 0

    def active_vars():
        out = set()
        for fr in frames:
            out |= fr.vars
        return out

    def active_floating(mandatory):
        out = []
        for fr in frames:
            for var in fr.f_order:
                if var in mandatory:
                    out.append(fr.f[var])
        return out

    def active_e():
        out = []
        for fr in frames:
            out.extend(fr.e)
        return out

    def active_dv():
        out = set()
        for fr in frames:
            out |= fr.dv
        return out

    def read_until(stop):
        nonlocal i
        out = []
        while i < len(toks) and toks[i] != stop:
            out.append(toks[i])
            i += 1
        i += 1
        return out

    while i < len(toks):
        tok = toks[i]
        i += 1
        if tok == "${":
            frames.append(edict(vars=set(), f={}, f_order=[], e=[], dv=set()))
        elif tok == "$}":
            frames.pop()
        elif tok == "$c":
            constants |= set(read_until("$."))
        elif tok == "$v":
            frames[-1].vars |= set(read_until("$."))
        elif tok == "$d":
            frames[-1].dv |= _pairs(read_until("$."))
        elif tok == "$[":
            read_until("$]")
        elif i < len(toks) and toks[i] in {"$f", "$e", "$a", "$p"}:
            label, kind = tok, toks[i]
            i += 1
            if kind == "$p":
                stmt = []
                while toks[i] != "$=":
                    stmt.append(toks[i])
                    i += 1
                i += 1
                read_until("$.")
                stmt = tuple(stmt)
            else:
                stmt = tuple(read_until("$."))
            labels[label] = tuple(stmt)
            if kind == "$f" and len(stmt) == 2:
                typecode, var = stmt
                frames[-1].f[var] = (label, typecode, var)
                frames[-1].f_order.append(var)
                var_type[var], var_flabel[var] = typecode, label
            elif kind == "$e":
                frames[-1].e.append((label, tuple(stmt)))
            elif kind in {"$a", "$p"} and stmt:
                avars = _vars_in(stmt, active_vars())
                essentials = tuple(e for _, e in active_e())
                mandatory = set(avars)
                for e in essentials:
                    mandatory |= _vars_in(e, active_vars())
                floating = tuple(active_floating(mandatory))
                rule = MMRule(
                    label=label,
                    conclusion=tuple(stmt),
                    floating=floating,
                    essential=essentials,
                    dv=frozenset(p for p in active_dv() if p[0] in mandatory and p[1] in mandatory),
                )
                rules[label] = rule
    return edict(labels=labels, rules=rules, var_type=var_type, var_flabel=var_flabel, constants=constants)


@lru_cache(maxsize=None)
def _rule_catalog(max_len=18):
    db = _database()
    out = []
    for r in db.rules.values():
        if not (1 <= len(r.essential) <= 3 and len(r.conclusion) <= max_len):
            continue
        if r.conclusion[0] != "|-" or any(len(h) > max_len for h in r.essential):
            continue
        if all(v in _VARS.get(t, ()) for _, t, v in r.floating):
            out.append(r)
    return out


@lru_cache(maxsize=None)
def _math_rule_catalog(max_len=18):
    return [r for r in _rule_catalog(max_len) if _keep_rule(r)]


def _balanced(xs):
    depth = 0
    for tok in xs:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _strip_outer(xs):
    xs = list(xs)
    while len(xs) >= 2 and xs[0] == "(" and xs[-1] == ")" and _balanced(xs[1:-1]):
        xs = xs[1:-1]
    return xs


def _find_top_level_binary(xs):
    for ops in _PRECEDENCE:
        depth = 0
        for i, tok in enumerate(xs):
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
            elif depth == 0 and tok in ops:
                return i, tok
    return None


def _display_token(tok, env=None):
    if env is not None:
        if tok in env.var_map:
            return env.var_map[tok]
        if tok in env.const_map:
            return env.const_map[tok]
    return _DISPLAY_VARS.get(tok, tok)


def _display_pred(op, env=None):
    if env is not None and op in env.pred_map:
        return env.pred_map[op]
    return op


def _display_func(op, env=None):
    if env is not None and op in env.func_map:
        return env.func_map[op]
    return op


def _is_context_var_expr(xs):
    return len(xs) == 1 and _database().var_type.get(xs[0]) == "wff"


def render_expr(expr, env=None):
    xs = _strip_outer(list(expr))
    if xs and xs[0] == "|-":
        return render_expr(xs[1:], env)
    xs = _strip_outer(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return _display_token(xs[0], env)
    if len(xs) == 2 and xs[0] in _PREFIX_APP:
        arg = render_expr([xs[1]], env)
        if arg is None:
            return None
        return f"{_display_func(xs[0], env)}({arg})"
    if len(xs) == 3 and xs[1] == "`":
        arg = render_expr([xs[2]], env)
        if arg is None:
            return None
        return f"{_display_func(xs[0], env)}({arg})"
    split = _find_top_level_binary(xs)
    if split:
        i, op = split
        left, right = render_expr(xs[:i], env), render_expr(xs[i + 1:], env)
        if left is None or right is None:
            return None
        if op == "->" and _is_context_var_expr(xs[:i]):
            return f"{left} => {right}"
        if op in _BIN_FUNC_TOKENS:
            return f"{_display_func(op, env)}({left}, {right})"
        return f"{_display_pred(op, env)}({left}, {right})"
    return None


def _math_score(expr):
    return len(set(expr) & _MATH_TOKENS)


def _has_math_atom(expr):
    return bool(set(expr) & _KEEP_HEADS)


def _wff_only(expr):
    return set(expr) <= _PURE_WFF | {"|-", "(", ")", "->", "<->", "/\\", "\\/"}


def _keep_rule(rule):
    exprs = [rule.conclusion, *rule.essential]
    rendered = [render_expr(e) for e in exprs]
    return (
        1 <= len(rule.essential) <= 3
        and all(r is not None and len(r) <= 120 for r in rendered)
        and (any(_has_math_atom(e) for e in exprs) or bool(set().union(*map(set, exprs)) & _KEEP_CONSTS))
        and sum(tok in _KEEP_HEADS for e in exprs for tok in e) >= 2
        and sum(_math_score(e) for e in exprs) >= 2
        and not all(_wff_only(e) for e in exprs)
    )


def _display_score(text):
    return len(re.findall(r"\b[PFDC]\d+\b", str(text)))


def _raw_relation_count(exprs):
    return sum(tok in _REL_TOKENS for expr in exprs for tok in expr)


def _raw_term_count(exprs):
    return sum(
        tok in _BIN_FUNC_TOKENS or tok in _PREFIX_APP or (i + 1 < len(expr) and expr[i + 1] == "`")
        for expr in exprs
        for i, tok in enumerate(expr)
    )


def _target_is_wff_implication(text):
    return bool(re.fullmatch(r"\(?[a-z]{2} => [a-z]{2}\)?", text.strip()))


def _display(text):
    if text is None:
        return None
    xs = text.strip()
    while xs.startswith("(") and xs.endswith(")") and _balanced(re.findall(r"\(|\)|[^()\s]+", xs[1:-1])):
        xs = xs[1:-1].strip()
    return xs


def _display_env(exprs):
    db = _database()
    seen_vars, seen_consts, seen_preds, seen_funcs = [], [], [], []
    syntax = {"|-", "(", ")", "`", "->"} | _REL_TOKENS | _LOGIC_PRED_TOKENS | _BIN_FUNC_TOKENS | set(_PREFIX_APP)
    for expr in exprs:
        xs = list(expr)
        for i, tok in enumerate(xs):
            if tok in db.var_type and tok not in seen_vars:
                seen_vars.append(tok)
            if tok in _REL_TOKENS | _LOGIC_PRED_TOKENS | {"->"} and tok not in seen_preds:
                seen_preds.append(tok)
            if tok in _BIN_FUNC_TOKENS and tok not in seen_funcs:
                seen_funcs.append(tok)
            if tok in _OBJECT_CONSTS and tok not in seen_consts:
                seen_consts.append(tok)
            if tok in _PREFIX_APP:
                if tok not in seen_funcs:
                    seen_funcs.append(tok)
            if i + 1 < len(xs) and xs[i + 1] == "`":
                if tok not in seen_funcs:
                    seen_funcs.append(tok)
            if (
                tok not in db.var_type and tok not in syntax and tok not in seen_consts
            ):
                seen_consts.append(tok)
    ctx_vars = [v for v in seen_vars if db.var_type.get(v) == "wff"]
    obj_vars = [v for v in seen_vars if db.var_type.get(v) != "wff"]
    object_const_names = {"0": "C0", "1": "C1", "2": "C2", "+oo": "C3", "-oo": "C4"}
    const_map, domain_i = {}, 1
    for c in seen_consts:
        if c in object_const_names:
            const_map[c] = object_const_names[c]
        else:
            const_map[c] = f"D{domain_i}"
            domain_i += 1
    return edict(
        var_map={
            **{v: _CONTEXT_NAMES[i % len(_CONTEXT_NAMES)] for i, v in enumerate(ctx_vars)},
            **{v: _OBJECT_NAMES[i % len(_OBJECT_NAMES)] for i, v in enumerate(obj_vars)},
        },
        const_map=const_map,
        pred_map={p: f"P{i + 1}" for i, p in enumerate(seen_preds)},
        func_map={f: f"F{i + 1}" for i, f in enumerate(seen_funcs)},
    )


def _subst_expr(expr, subst):
    out = []
    for tok in expr:
        out.extend(subst.get(tok, (tok,)))
    return tuple(out)


def _fresh_subst(rule):
    used = defaultdict(int)
    subst = {}
    for _, typecode, var in rule.floating:
        pool = _VARS.get(typecode, (var,))
        subst[var] = (pool[used[typecode] % len(pool)],)
        used[typecode] += 1
    for a, b in rule.dv:
        if subst.get(a) == subst.get(b):
            return None
    return subst


def _head(expr):
    body = expr[1:] if expr and expr[0] in _TYPECODES else expr
    return next((t for t in body if t not in {"(", ")"} and t not in _database().var_type), "")


def _unify_variable_only(pattern, target, subst=None):
    db = _database()
    if len(pattern) != len(target):
        return None
    subst = dict(subst or {})
    for p, t in zip(pattern, target):
        if p in db.var_type:
            if t not in db.var_type or db.var_type[t] != db.var_type[p]:
                return None
            if p in subst and subst[p] != (t,):
                return None
            subst[p] = (t,)
        elif p != t:
            return None
    return subst


def _match_variable_only(pattern, target):
    return _unify_variable_only(pattern, target, {})


def _rules_matching_variable_only(target, rules):
    random.shuffle(rules)
    return [r for r in rules if _match_variable_only(r.conclusion, target) is not None]


def _local_f_label(var):
    return _database().var_flabel[var]


def _build_tree(rule, rules, depth, leaf_prob=0.45, target=None):
    subst = _match_variable_only(rule.conclusion, target) if target is not None else _fresh_subst(rule)
    if subst is None:
        return None
    used_by_type = defaultdict(set)
    for _, typecode, var in rule.floating:
        if var in subst:
            used_by_type[typecode].add(subst[var][0])
    for _, typecode, var in rule.floating:
        if var in subst:
            continue
        pool = [v for v in _VARS.get(typecode, (var,)) if v not in used_by_type[typecode]]
        if not pool:
            pool = list(_VARS.get(typecode, (var,)))
        subst[var] = (pool[0],)
        used_by_type[typecode].add(pool[0])
    for a, b in rule.dv:
        if subst.get(a) == subst.get(b):
            return None
    target = _subst_expr(rule.conclusion, subst)
    leaves, proof, used = [], [], {rule.label}
    proof.extend(_local_f_label(v[0]) for _, _, v0 in rule.floating if (v := subst[v0]) and v[0] in _database().var_flabel)
    for hyp in rule.essential:
        shyp = _subst_expr(hyp, subst)
        child = None
        if depth > 0 and random.random() >= leaf_prob:
            matches = [
                r for r in _rules_matching_variable_only(shyp, list(rules))
                if r.label != rule.label or random.random() < 0.35
            ]
            if matches:
                child = _build_tree(random.choice(matches), rules, depth - 1, leaf_prob, target=shyp)
        if child is None:
            leaves.append(shyp)
            proof.append("__LEAF__")
        else:
            leaves.extend(child.leaves)
            proof.extend(child.proof)
            used |= set(child.used)
    proof.append(rule.label)
    return MMTree(target, tuple(leaves), tuple(proof), frozenset(used))


def _verify(proof, target, premises, allowed_rules, check_dv=True):
    db = _database()
    labels = {k: db.labels[k] for k in db.var_flabel.values() if k in db.labels}
    rules = {k: db.rules[k] for k in allowed_rules if k in db.rules}
    for i, p in enumerate(premises, 1):
        labels[f"prem{i}"] = tuple(p)
    stack = []
    for label in proof:
        if label in labels:
            stack.append(labels[label])
            continue
        rule = rules[label]
        subst = {}
        hyps = [tuple((t, v)) for _, t, v in rule.floating] + list(rule.essential)
        if len(stack) < len(hyps):
            return False
        args = stack[-len(hyps):] if hyps else []
        if hyps:
            del stack[-len(hyps):]
        for (_, t, v), arg in zip(rule.floating, args[:len(rule.floating)]):
            if not arg or arg[0] != t:
                return False
            subst[v] = tuple(arg[1:])
        for hyp, arg in zip(rule.essential, args[len(rule.floating):]):
            if _subst_expr(hyp, subst) != tuple(arg):
                return False
        if check_dv:
            for a, b in rule.dv:
                va, vb = subst.get(a, ()), subst.get(b, ())
                if va == vb or set(va) & set(vb):
                    return False
        stack.append(_subst_expr(rule.conclusion, subst))
    return stack == [tuple(target)]


def _tree_with_premise_labels(tree):
    mapping, premises = {}, []
    for leaf in tree.leaves:
        if leaf not in mapping:
            mapping[leaf] = f"prem{len(mapping) + 1}"
            premises.append(leaf)
    proof, leaf_iter = [], iter(tree.leaves)
    for label in tree.proof:
        if label == "__LEAF__":
            proof.append(mapping[next(leaf_iter)])
        else:
            proof.append(label)
    return tuple(premises), tuple(proof)


def _variable_only_subst(rule, subst, check_dv=True):
    db = _database()
    for _, typecode, var in rule.floating:
        val = subst.get(var)
        if not val or len(val) != 1 or db.var_type.get(val[0]) != typecode:
            return False
    return not check_dv or all(subst.get(a) != subst.get(b) for a, b in rule.dv)


def _closure_proof(rule, subst, args, known):
    proof = []
    for _, _, var in rule.floating:
        proof.append(_local_f_label(subst[var][0]))
    for arg in args:
        proof.extend(known[tuple(arg)])
    proof.append(rule.label)
    return tuple(proof)


def _closure(premises, rules, cap=18, check_dv=True):
    labels = tuple(sorted(r.label for r in rules))
    return _closure_cached(tuple(map(tuple, premises)), labels, int(cap), bool(check_dv))


def _relaxed_gap(real_closure, relaxed_closure, target, kind):
    if real_closure is not None and relaxed_closure is not None:
        if target not in real_closure and target in relaxed_closure:
            _DV_RELAX_STATS[f"{kind}_real_absent_relaxed_present"] += 1
            return True
    return False


def dv_relax_stats():
    return dict(_DV_RELAX_STATS)


@lru_cache(maxsize=512)
def _closure_cached(premises, rule_labels, cap, check_dv):
    db = _database()
    rules = [db.rules[l] for l in rule_labels]
    known = {tuple(p): (f"prem{i + 1}",) for i, p in enumerate(premises)}
    by_head = defaultdict(list)
    for formula in known:
        by_head[_head(formula)].append(formula)
    changed = True
    matches = 0
    while changed:
        changed = False
        known_items = list(known)
        for rule in rules:
            if not rule.essential:
                continue
            pools = [
                by_head.get(_head(hyp), known_items) or known_items
                for hyp in rule.essential
            ]
            for args in itertools.product(*pools):
                matches += 1
                if matches > _MAX_CLOSURE_MATCHES:
                    return None
                subst = {}
                for hyp, arg in zip(rule.essential, args):
                    subst = _unify_variable_only(hyp, arg, subst)
                    if subst is None:
                        break
                if subst is None or not _variable_only_subst(rule, subst, check_dv=check_dv):
                    continue
                conc = _subst_expr(rule.conclusion, subst)
                if len(conc) > cap or conc in known:
                    continue
                proof = _closure_proof(rule, subst, args, known)
                if not _verify(proof, conc, premises, rule_labels, check_dv=check_dv):
                    continue
                known[conc] = proof
                by_head[_head(conc)].append(conc)
                if len(known) > _MAX_CLOSURE_FORMULAS:
                    return None
                changed = True
    return known


def _fmt(expr, env=None):
    return _display(render_expr(expr, env))


def _rule_schema(label, env=None):
    rule = _database().rules[label]
    lhs = "; ".join(_fmt(h, env) for h in rule.essential)
    return f"{lhs} ==> {_fmt(rule.conclusion, env)}"


def _rule_rows(labels, env=None):
    return [
        edict(id=f"r{i + 1}", label=label, schema=_rule_schema(label, env))
        for i, label in enumerate(labels)
    ]


def _rule_text(rows, bullet=False):
    prefix = "- " if bullet else ""
    return "\n".join(f"{prefix}{r.id}: {r.schema}" for r in rows)


def _rule_ids(labels, rows):
    ids = {r.label: r.id for r in rows}
    return [ids[label] for label in labels]


def _keep_instance(inst):
    display_exprs = list(inst.premises) + [inst.tree.target]
    for label in inst.rules:
        rule = _database().rules[label]
        display_exprs.extend(rule.essential)
        display_exprs.append(rule.conclusion)
    env = _display_env(display_exprs)
    rendered_premises = [_fmt(p, env) for p in inst.premises]
    rendered_target = _fmt(inst.tree.target, env)
    rendered_rules = [_rule_schema(label, env) for label in inst.rules]
    if rendered_target is None or any(p is None for p in rendered_premises + rendered_rules):
        return False
    exprs = [inst.tree.target, *inst.premises]
    text = "\n".join([rendered_target, *rendered_premises])
    return (
        len(inst.premises) >= 2
        and len(inst.tree.used) >= 2
        and _raw_relation_count([inst.tree.target]) >= 1
        and _raw_relation_count(exprs) >= 3
        and _display_score(text) >= 3
        and not _target_is_wff_implication(rendered_target)
    )


def _sample_instance(config):
    rules = _math_rule_catalog(int(config.formula_len_cap))
    if not rules:
        raise RuntimeError("No small Metamath rules mined from set.mm")
    for _ in range(200):
        root = random.choice(rules)
        tree = _build_tree(root, rules, int(config.proof_depth))
        if (
            not tree or len(tree.used) < 2 or len(tree.used) > int(config.n_rules)
            or not tree.leaves or len(set(tree.leaves)) > int(config.max_premises)
        ):
            continue
        premises, proof = _tree_with_premise_labels(tree)
        if not _verify(proof, tree.target, premises, tree.used):
            continue
        displayed = sorted(tree.used)
        # EasyDict releases differ in whether nested tuples are preserved.  These
        # proof expressions rely on tuple concatenation, so keep this internal
        # container out of EasyDict's recursive conversion path.
        inst = SimpleNamespace(
            tree=tree, premises=premises, proof=proof, rules=displayed
        )
        if _keep_instance(inst):
            return inst
    raise RuntimeError("failed to generate a verified Metamath proof tree")


def _critical_premise_foil(inst, config):
    """Permute same-typed variables while preserving global token counts."""
    rules = [_database().rules[l] for l in inst.rules]
    cap = int(config.formula_len_cap)
    db = _database()
    full = _closure(inst.premises, rules, cap, check_dv=True)
    if full is None or inst.tree.target not in full:
        return None
    order = list(range(len(inst.premises)))
    random.shuffle(order)
    for i in order:
        premise = inst.premises[i]
        without = inst.premises[:i] + inst.premises[i + 1:]
        relaxed = _closure(without, rules, cap, check_dv=False)
        if relaxed is None or inst.tree.target in relaxed:
            continue
        positions = [j for j, tok in enumerate(premise) if tok in db.var_type]
        swaps = [
            (j, k) for j, k in itertools.combinations(positions, 2)
            if db.var_type[premise[j]] == db.var_type[premise[k]]
            and premise[j] != premise[k]
        ]
        random.shuffle(swaps)
        candidates = []
        for j, k in swaps:
            foil = list(premise)
            foil[j], foil[k] = foil[k], foil[j]
            candidates.append((i, tuple(foil), None, None))

        # If one formula has no useful internal permutation, exchange two
        # same-typed tokens across premises. This changes structure but keeps
        # the entire prompt's variable multiset exactly fixed.
        for k, other in enumerate(inst.premises):
            if k == i:
                continue
            cross = [
                (j, ell) for j in positions
                for ell, tok in enumerate(other)
                if tok in db.var_type
                and db.var_type[premise[j]] == db.var_type[tok]
                and premise[j] != tok
                and premise.count(premise[j]) == 1
                and other.count(tok) == 1
                and tok not in premise
                and premise[j] not in other
            ]
            random.shuffle(cross)
            for j, ell in cross[:8]:
                left, right = list(premise), list(other)
                left[j], right[ell] = right[ell], left[j]
                candidates.append((i, tuple(left), k, tuple(right)))

        random.shuffle(candidates)
        for first_i, first, second_i, second in candidates:
            negative = list(inst.premises)
            negative[first_i] = first
            if second_i is not None:
                negative[second_i] = second
            negative = tuple(negative)
            if len(set(negative)) != len(negative) or any(_fmt(p) is None for p in negative):
                continue
            closure = _closure(negative, rules, cap, check_dv=False)
            if closure is not None and inst.tree.target not in closure:
                changed = [first_i] + ([] if second_i is None else [second_i])
                return negative, tuple(changed)
    return None


def _abc(i):
    return chr(ord("A") + i)


class MetamathEntailment(Task):
    """Choose which token-balanced premise set entails a conjecture."""
    summary = "Choose which structurally matched premise set derives a conjecture."
    task_version = 2

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or MetamathConfig(), timeout=120, **kwargs)

    def generate_entry(self):
        for _ in range(50):
            inst = _sample_instance(self.config)
            # Polarity must be chosen per-call (stateless): the old instance-state
            # toggle (self._want_positive) degenerates under generate_balanced_batch(workers>1),
            # where each pool.submit re-pickles a fresh `self` from the parent so the toggle
            # never advances -> every worker emits the same polarity -> balancing cap livelock.
            pair = _critical_premise_foil(inst, self.config)
            if pair is None:
                continue
            negative, changed = pair
            premise_sets = [inst.premises, negative]
            random.shuffle(premise_sets)
            answer = "AB"[premise_sets.index(inst.premises)]
            target = inst.tree.target
            display_exprs = [p for group in premise_sets for p in group] + [target]
            for label in inst.rules:
                rule = _database().rules[label]
                display_exprs.extend(rule.essential)
                display_exprs.append(rule.conclusion)
            env = _display_env(display_exprs)
            rule_rows = _rule_rows(inst.rules, env)
            displayed_sets = [
                [f"{i + 1}. {_fmt(p, env)}" for i, p in enumerate(group)]
                for group in premise_sets
            ]
            meta = edict(
                premise_sets=[[_fmt(p, env) for p in group] for group in premise_sets],
                raw_premise_sets=[[list(p) for p in group] for group in premise_sets],
                rules=[r.id for r in rule_rows],
                raw_rule_labels=inst.rules,
                rule_map={r.id: r.label for r in rule_rows},
                rule_schemas={r.id: r.schema for r in rule_rows},
                anonymization=dict(
                    var_map=env.var_map,
                    const_map=env.const_map,
                    pred_map=env.pred_map,
                    func_map=env.func_map,
                ),
                dv_relax_stats=dv_relax_stats(),
                conjecture=_fmt(target, env),
                raw_conjecture=list(target),
                sufficient_option=answer,
                changed_premises=[i + 1 for i in changed],
                token_counts_balanced=(Counter(itertools.chain.from_iterable(premise_sets[0])) ==
                                       Counter(itertools.chain.from_iterable(premise_sets[1]))),
                proof=list(inst.proof),
                source="set.mm",
            )
            meta.payload = {
                "premise_set_a": "\n".join(displayed_sets[0]),
                "premise_set_b": "\n".join(displayed_sets[1]),
                "allowed_rules": _rule_text(rule_rows),
                "conjecture": meta.conjecture,
            }
            return Entry(meta, answer)
        raise RuntimeError("failed to generate Metamath entailment task")

    def render_prompt(self, metadata):
        return (
            "Which premise set makes the conjecture follow using only the listed rules?\n"
            "Exactly one of A and B is sufficient.\n"
            "Rules instantiate only by renaming variables.\n"
            "The answer is A or B.\n\n"
            f"{render_payload(metadata.payload)}"
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip().strip("`").upper() == entry.answer)


class MetamathCoreSelect(Task):
    """Select the minimal sufficient displayed rule subset for a Metamath proof."""

    summary = "Select the minimal sufficient displayed Metamath rule subset for a generated formal proof obligation."

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or MetamathConfig(), timeout=120, **kwargs)

    def generate_entry(self):
        labels = [r.label for r in _math_rule_catalog(int(self.config.formula_len_cap))]
        for _ in range(80):
            inst = _sample_instance(self.config)
            core = tuple(sorted(inst.tree.used))
            if len(core) < 2:
                continue
            core_closure = _closure(
                inst.premises,
                [_database().rules[l] for l in core if l in _database().rules],
                int(self.config.formula_len_cap),
                check_dv=True,
            )
            if core_closure is not None and inst.tree.target in core_closure:
                pass
            else:
                continue
            minimal = True
            for miss in core:
                minus = _closure(
                    inst.premises,
                    [_database().rules[l] for l in core if l != miss],
                    int(self.config.formula_len_cap),
                    check_dv=False,
                )
                if minus is None or inst.tree.target in minus:
                    minimal = False
                    break
            if not minimal:
                continue
            wrong = set()
            if len(core) > 1:
                wrong.add(tuple(x for x in core if x != random.choice(core)))
            near = [l for l in labels if l not in core]
            random.shuffle(near)
            for label in near:
                cand = tuple(sorted((set(core) - {random.choice(core)}) | {label}))
                cand_closure = _closure(
                    inst.premises,
                    [_database().rules[l] for l in cand],
                    int(self.config.formula_len_cap),
                    check_dv=True,
                )
                cand_relaxed = _closure(
                    inst.premises,
                    [_database().rules[l] for l in cand],
                    int(self.config.formula_len_cap),
                    check_dv=False,
                )
                if cand_closure is not None and cand_relaxed is not None:
                    _relaxed_gap(cand_closure, cand_relaxed, inst.tree.target, "wrong_option")
                if (
                    cand != core and cand not in wrong
                    and cand_closure is not None and cand_relaxed is not None
                    and inst.tree.target not in cand_closure
                    and inst.tree.target not in cand_relaxed
                ):
                    wrong.add(cand)
                if len(wrong) >= int(self.config.n_options) - 1:
                    break
            if len(wrong) < int(self.config.n_options) - 1:
                continue
            options = [core] + list(wrong)[: int(self.config.n_options) - 1]
            if len(set(options)) != int(self.config.n_options) or any(not o for o in options):
                continue
            random.shuffle(options)
            answer = _abc(options.index(core))
            displayed_labels = sorted(set().union(*map(set, options)))
            display_exprs = list(inst.premises) + [inst.tree.target]
            for label in displayed_labels:
                rule = _database().rules[label]
                display_exprs.extend(rule.essential)
                display_exprs.append(rule.conclusion)
            env = _display_env(display_exprs)
            rule_rows = _rule_rows(displayed_labels, env)
            premises = [f"{i + 1}. {_fmt(p, env)}" for i, p in enumerate(inst.premises)]
            option_text = "\n".join(
                f"{_abc(i)}. [{', '.join(_rule_ids(o, rule_rows))}]"
                for i, o in enumerate(options)
            )
            meta = edict(
                premises=[_fmt(p, env) for p in inst.premises],
                raw_premises=[list(p) for p in inst.premises],
                conjecture=_fmt(inst.tree.target, env),
                raw_conjecture=list(inst.tree.target),
                rule_map={r.id: r.label for r in rule_rows},
                rule_schemas={r.id: r.schema for r in rule_rows},
                anonymization=dict(
                    var_map=env.var_map,
                    const_map=env.const_map,
                    pred_map=env.pred_map,
                    func_map=env.func_map,
                ),
                dv_relax_stats=dv_relax_stats(),
                options=[_rule_ids(o, rule_rows) for o in options],
                raw_options=[list(o) for o in options],
                proof=list(inst.proof),
                source="set.mm",
            )
            meta.payload = {
                "premises": "\n".join(premises),
                "rule_catalog": _rule_text(rule_rows, bullet=True),
                "conjecture": meta.conjecture,
                "options": option_text,
            }
            return Entry(
                meta,
                answer,
            )
        raise RuntimeError("failed to generate Metamath core-selection task")

    def render_prompt(self, metadata):
        return (
            "Which option is sufficient to derive the conjecture?\n"
            "Use only the listed premises and rules. No hidden background facts.\n"
            "Rules may only rename variables, not substitute compound terms.\n"
            "The answer is A, B, C, or D.\n\n"
            f"{render_payload(metadata.payload)}"
        )

    def score_answer(self, answer, entry):
        m = re.search(r"[A-D]", str(answer).upper())
        return float(bool(m) and m.group(0) == entry.answer)
