"""Procedural first-order rewrite tasks with compound substitution.

The visible tasks share one hidden generator:

- ``MathScratchEntailment`` asks whether two terms have the same normal form.
- ``MathScratchCoreSelect`` asks which displayed rule subset is sufficient.
- ``MathScratchNormalize`` asks for the normal form of one term.

By default rules are syntactically first-order and strictly size-decreasing
under every substitution. ``rule_mode="lpo"`` enables richer structure-moving
rules oriented by a sampled lexicographic path order.
"""

from __future__ import annotations

import random
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

from easydict import EasyDict as edict

from reasoning_core.template import Config, DevTask, Entry, Task, render_payload, stochastic_rounding as sround


Term = Any
Position = tuple[int, ...]

_VARS = ("?X", "?Y", "?Z")


@dataclass
class ScratchConfig(Config):
    rule_mode: str = "size_decreasing"
    n_rules: int = 6
    n_options: int = 4
    n_consts: int = 4
    n_unary: int = 3
    n_binary: int = 2
    min_steps: int = 2
    max_steps: int = 6
    min_core_size: int = 2
    max_core_size: int = 4
    max_term_depth: int = 4
    max_rule_lhs_depth: int = 3
    max_norm_steps: int = 64
    max_intermediate_size: int = 512
    max_tries: int = 250
    guard_samples: int = 24
    random_strategy_trials: int = 4
    rhs_size_tolerance: int = 3
    world_cache_examples: int = 16
    surface_match_negatives: bool = True

    def apply_difficulty(self, level):
        self.n_rules = sround(self.n_rules + level / 2)
        self.min_steps = sround(self.min_steps + level / 2)
        self.max_steps = sround(self.max_steps + level)
        self.max_term_depth = sround(self.max_term_depth + level)
        self.max_rule_lhs_depth = sround(self.max_rule_lhs_depth + level / 2)
        self.max_core_size = min(6, sround(self.max_core_size + level / 3))


@dataclass(frozen=True)
class ScratchRule:
    name: str
    lhs: Term
    rhs: Term
    mode: str = "size_decreasing"
    precedence: tuple[str, ...] = ()


@dataclass
class ScratchInstance:
    rules: list[ScratchRule]
    lhs: Term
    rhs: Term
    used: tuple[str, ...]
    proof: list[tuple[Term, str, Position, Term]]


@dataclass
class ScratchWorld:
    rules: list[ScratchRule]
    world_id: str
    diagnostics: edict


def _is_var(term):
    return isinstance(term, str) and term.startswith("?")


def _is_app(term):
    return isinstance(term, tuple)


def _args(term):
    return term[1:] if _is_app(term) else ()


def _size(term):
    if not _is_app(term):
        return 1
    return 1 + sum(_size(a) for a in _args(term))


def _depth(term):
    if not _is_app(term):
        return 0
    return 1 + max((_depth(a) for a in _args(term)), default=0)


def _vars(term):
    if _is_var(term):
        return {term}
    if _is_app(term):
        out = set()
        for arg in _args(term):
            out |= _vars(arg)
        return out
    return set()


def _var_counts(term):
    if _is_var(term):
        return Counter([term])
    if _is_app(term):
        out = Counter()
        for arg in _args(term):
            out.update(_var_counts(arg))
        return out
    return Counter()


def _contains_function(term):
    return _is_app(term) or any(_contains_function(a) for a in _args(term))


def _render(term):
    if _is_var(term):
        return term[1:]
    if not _is_app(term):
        return str(term)
    return f"{term[0]}({', '.join(_render(a) for a in _args(term))})"


def _term_data(term):
    if _is_app(term):
        return [term[0], *[_term_data(a) for a in _args(term)]]
    return term


def _subterms(term):
    out = [term]
    if _is_app(term):
        for arg in _args(term):
            out.extend(_subterms(arg))
    return out


def _symbols(term):
    if _is_var(term):
        return set()
    if _is_app(term):
        out = {term[0]}
        for arg in _args(term):
            out |= _symbols(arg)
        return out
    return {term}


def _precedence(config):
    symbols = _constants(config) + _unary_functions(config) + _binary_functions(config)
    random.shuffle(symbols)
    return tuple(symbols)


def _prec_map(precedence):
    return {symbol: len(precedence) - i for i, symbol in enumerate(precedence)}


def _head(term):
    if _is_var(term):
        return None
    return term[0] if _is_app(term) else term


def _lpo_ge(lhs, rhs, prec):
    return lhs == rhs or _lpo_gt(lhs, rhs, prec)


def _lpo_gt(lhs, rhs, prec):
    if lhs == rhs or _is_var(lhs):
        return False
    if any(_lpo_ge(arg, rhs, prec) for arg in _args(lhs)):
        return True
    if _is_var(rhs):
        return rhs in _vars(lhs)
    if not all(_lpo_gt(lhs, arg, prec) for arg in _args(rhs)):
        return False
    left_head, right_head = _head(lhs), _head(rhs)
    if left_head != right_head:
        return prec.get(left_head, 0) > prec.get(right_head, 0)
    left_args, right_args = _args(lhs), _args(rhs)
    for left_arg, right_arg in zip(left_args, right_args):
        if left_arg == right_arg:
            continue
        return _lpo_gt(left_arg, right_arg, prec)
    return len(left_args) > len(right_args)


def _rule_decreases(rule, before, after):
    if rule.mode == "lpo":
        return _lpo_gt(before, after, _prec_map(rule.precedence))
    return _size(after) < _size(before)


def _valid_rule(lhs, rhs, config=None, precedence=()):
    left_counts = _var_counts(lhs)
    right_counts = _var_counts(rhs)
    mode = getattr(config, "rule_mode", "size_decreasing") if config is not None else "size_decreasing"
    common = (
        not _is_var(lhs)
        and _contains_function(lhs)
        and set(right_counts) <= set(left_counts)
    )
    if mode == "lpo":
        return common and _lpo_gt(lhs, rhs, _prec_map(precedence))
    return (
        common
        and all(right_counts[v] <= left_counts[v] for v in right_counts)
        and _size(rhs) < _size(lhs)
    )


def _match(pattern, term, env=None):
    env = dict(env or {})
    if _is_var(pattern):
        bound = env.get(pattern)
        if bound is None:
            env[pattern] = term
            return env
        return env if bound == term else None
    if not _is_app(pattern):
        return env if pattern == term else None
    if not _is_app(term) or pattern[0] != term[0] or len(pattern) != len(term):
        return None
    for p_arg, t_arg in zip(_args(pattern), _args(term)):
        env = _match(p_arg, t_arg, env)
        if env is None:
            return None
    return env


def _subst(pattern, env):
    if _is_var(pattern):
        return env[pattern]
    if _is_app(pattern):
        return (pattern[0], *(_subst(a, env) for a in _args(pattern)))
    return pattern


def _positions(term):
    yield ()
    if _is_app(term):
        for i, arg in enumerate(_args(term)):
            for pos in _positions(arg):
                yield (i,) + pos


def _get_at(term, pos):
    for i in pos:
        term = _args(term)[i]
    return term


def _set_at(term, pos, value):
    if not pos:
        return value
    if not _is_app(term):
        raise ValueError("position descends into atom")
    i = pos[0]
    args = list(_args(term))
    args[i] = _set_at(args[i], pos[1:], value)
    return (term[0], *args)


def _rewrite_once(term, rules):
    for pos in _positions(term):
        sub = _get_at(term, pos)
        matching = [rule for rule in rules if _match(rule.lhs, sub) is not None]
        if not matching:
            continue
        rule = matching[0]
        env = _match(rule.lhs, sub)
        return _set_at(term, pos, _subst(rule.rhs, env)), rule.name, pos, rule
    return None


def _all_rewrites(term, rules):
    out = []
    for pos in _positions(term):
        sub = _get_at(term, pos)
        for rule in rules:
            env = _match(rule.lhs, sub)
            if env is not None:
                out.append((_set_at(term, pos, _subst(rule.rhs, env)), rule.name, pos, rule))
    return out


def _normalize(term, rules, max_steps=64, random_strategy=False, rng=None, max_size=512):
    rng = rng or random
    trace = []
    for _ in range(int(max_steps)):
        if random_strategy:
            candidates = _all_rewrites(term, rules)
            step = rng.choice(candidates) if candidates else None
        else:
            step = _rewrite_once(term, rules)
        if step is None:
            return term, trace
        new_term, rule_name, pos, rule = step
        trace.append((term, rule_name, pos, new_term))
        if not _rule_decreases(rule, term, new_term):
            raise RuntimeError("rewrite did not decrease under the configured order")
        if _size(new_term) > int(max_size):
            raise RuntimeError("normalization exceeded intermediate size limit")
        term = new_term
    raise RuntimeError("normalization exceeded step limit")


def _equivalent(lhs, rhs, rules, config):
    left_nf, _ = _normalize(lhs, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
    right_nf, _ = _normalize(rhs, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
    return left_nf == right_nf


def _random_strategy_equivalent(lhs, rhs, rules, config):
    for _ in range(int(config.random_strategy_trials)):
        left_nf, _ = _normalize(
            lhs, rules, int(config.max_norm_steps), random_strategy=True, max_size=int(config.max_intermediate_size)
        )
        right_nf, _ = _normalize(
            rhs, rules, int(config.max_norm_steps), random_strategy=True, max_size=int(config.max_intermediate_size)
        )
        if left_nf != right_nf:
            return False
    return True


def _certify_equivalent(lhs, rhs, rules, config):
    return _equivalent(lhs, rhs, rules, config) and _random_strategy_equivalent(lhs, rhs, rules, config)


def _certify_inequivalent(lhs, rhs, rules, config):
    if _equivalent(lhs, rhs, rules, config):
        return False
    for _ in range(int(config.random_strategy_trials)):
        left_nf, _ = _normalize(
            lhs, rules, int(config.max_norm_steps), random_strategy=True, max_size=int(config.max_intermediate_size)
        )
        right_nf, _ = _normalize(
            rhs, rules, int(config.max_norm_steps), random_strategy=True, max_size=int(config.max_intermediate_size)
        )
        if left_nf == right_nf:
            return False
    return True


def _constants(config):
    return [f"c{i}" for i in range(int(config.n_consts))]


def _unary_functions(config):
    return [f"f{i}" for i in range(int(config.n_unary))]


def _binary_functions(config):
    return [f"g{i}" for i in range(int(config.n_binary))]


def _sample_ground_term(config, depth=None):
    depth = int(config.max_term_depth if depth is None else depth)
    if depth <= 0 or random.random() < 0.35:
        return random.choice(_constants(config))
    if random.random() < 0.55:
        return (random.choice(_unary_functions(config)), _sample_ground_term(config, depth - 1))
    return (
        random.choice(_binary_functions(config)),
        _sample_ground_term(config, depth - 1),
        _sample_ground_term(config, depth - 1),
    )


def _sample_pattern_term(config, depth, variables=_VARS, var_prob=0.55):
    if depth <= 0 or random.random() < 0.30:
        if random.random() < var_prob:
            return random.choice(variables)
        return random.choice(_constants(config))
    if random.random() < 0.55:
        return (random.choice(_unary_functions(config)), _sample_pattern_term(config, depth - 1, variables, var_prob))
    return (
        random.choice(_binary_functions(config)),
        _sample_pattern_term(config, depth - 1, variables, var_prob),
        _sample_pattern_term(config, depth - 1, variables, var_prob),
    )


def _sample_rule_candidate(config):
    x, y = "?X", "?Y"
    const = random.choice(_constants(config))
    unary = random.choice(_unary_functions(config))
    binary = random.choice(_binary_functions(config))
    kind = random.randrange(8)
    if kind == 0:
        return (unary, x), x
    if kind == 1:
        return (unary, (random.choice(_unary_functions(config)), x)), x
    if kind == 2:
        return (binary, x, const), x
    if kind == 3:
        return (binary, const, x), x
    if kind == 4:
        return (binary, x, x), x
    if kind == 5:
        return (unary, const), random.choice(_constants(config))
    lhs = _sample_pattern_term(config, random.randint(1, int(config.max_rule_lhs_depth)))
    proper = [t for t in _subterms(lhs) if t != lhs and _vars(t) <= _vars(lhs)]
    if proper and random.random() < 0.75:
        rhs = random.choice(proper)
    else:
        allowed_vars = tuple(_vars(lhs)) or (x,)
        rhs = _sample_pattern_term(config, max(0, _depth(lhs) - 1), allowed_vars, var_prob=0.75)
    return lhs, rhs


def _sample_lpo_rule_candidate(config):
    x, y, z = "?X", "?Y", "?Z"
    unary = random.choice(_unary_functions(config))
    unary2 = random.choice(_unary_functions(config))
    binary = random.choice(_binary_functions(config))
    binary2 = random.choice(_binary_functions(config))
    const = random.choice(_constants(config))
    kind = random.randrange(10)
    if kind == 0:
        return (unary, (binary, x, y)), (binary, (unary, x), (unary, y))
    if kind == 1:
        return (unary, (binary, x, y)), (binary, (unary, x), y)
    if kind == 2:
        return (binary, x, (binary2, y, z)), (binary2, (binary, x, y), (binary, x, z))
    if kind == 3:
        return (binary, (unary, x), (unary, y)), (unary, (binary, x, y))
    if kind == 4:
        return (binary, (binary2, x, y), z), (binary2, x, (binary, y, z))
    if kind == 5:
        return (unary, (unary2, x)), (unary2, (unary, x))
    if kind == 6:
        return (binary, x, const), (unary, x)
    return _sample_rule_candidate(config)


def _sample_rules(config):
    rules = []
    seen = set()
    precedence = _precedence(config)
    for _ in range(int(config.max_tries) * 4):
        lhs, rhs = _sample_lpo_rule_candidate(config) if config.rule_mode == "lpo" else _sample_rule_candidate(config)
        key = (_render(lhs), _render(rhs))
        if key in seen or not _valid_rule(lhs, rhs, config, precedence):
            continue
        seen.add(key)
        rules.append(ScratchRule(f"r{len(rules) + 1}", lhs, rhs, config.rule_mode, precedence))
        if len(rules) >= int(config.n_rules):
            return rules
    raise RuntimeError("failed to sample enough valid scratch rewrite rules")


def _nondegenerate(rules, config):
    samples = [_sample_ground_term(config, int(config.max_term_depth)) for _ in range(max(24, int(config.guard_samples)))]
    nfs, fired = [], Counter()
    for term in samples:
        try:
            nf, trace = _normalize(term, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
        except RuntimeError:
            return False
        nfs.append(nf)
        fired.update(rule for _, rule, _, _ in trace)
        for _ in range(int(config.random_strategy_trials)):
            try:
                rnf, _ = _normalize(
                    term, rules, int(config.max_norm_steps), random_strategy=True, max_size=int(config.max_intermediate_size)
                )
            except RuntimeError:
                return False
            if rnf != nf:
                return False
    counts = Counter(_render(t) for t in nfs)
    if len(counts) < max(3, len(samples) // 4):
        return False
    if counts.most_common(1)[0][1] > len(samples) * 0.5:
        return False
    return len(fired) >= max(2, int(len(rules) * 0.5))


def _sample_world(config):
    for _ in range(int(config.max_tries)):
        rules = _sample_rules(config)
        if _nondegenerate(rules, config):
            return rules
    raise RuntimeError("failed to sample a nondegenerate scratch rewrite world")


def _fill_missing_env(lhs, env, config):
    env = dict(env)
    for var in sorted(_vars(lhs)):
        if var not in env:
            env[var] = _sample_ground_term(config, random.randint(0, max(0, int(config.max_term_depth) - 2)))
    return env


def _anti_step(term, rules, config, used):
    positions = list(_positions(term))
    random.shuffle(positions)
    preferred = [rule for rule in rules if rule.name not in used]
    fallback = [rule for rule in rules if rule.name in used]
    random.shuffle(preferred)
    random.shuffle(fallback)
    candidates = preferred + fallback
    for pos in positions:
        subterm = _get_at(term, pos)
        for rule in candidates:
            env = _match(rule.rhs, subterm)
            if env is None:
                continue
            env = _fill_missing_env(rule.lhs, env, config)
            expanded_subterm = _subst(rule.lhs, env)
            expanded = _set_at(term, pos, expanded_subterm)
            if expanded == term:
                continue
            if rule.mode != "lpo" and _size(expanded) <= _size(term):
                continue
            if _depth(expanded) > int(config.max_term_depth) + int(config.max_steps):
                continue
            if _size(expanded) > int(config.max_intermediate_size):
                continue
            return expanded, rule.name, pos
    return None


def _rule_map(rules):
    return {rule.name: rule for rule in rules}


def _ordered(names, rules):
    name_set = set(names)
    return [rule for rule in rules if rule.name in name_set]


def _sample_positive_instance(rules, config):
    for _ in range(int(config.max_tries)):
        right_seed = _sample_ground_term(config, random.randint(0, max(1, int(config.max_term_depth) // 2)))
        right, _ = _normalize(right_seed, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
        left = right
        planned = []
        for _ in range(random.randint(int(config.min_steps), int(config.max_steps))):
            step = _anti_step(left, rules, config, set(planned))
            if step is None:
                break
            left, rule_name, _ = step
            planned.append(rule_name)
        try:
            nf, trace = _normalize(left, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
        except RuntimeError:
            continue
        used = tuple(dict.fromkeys(rule for _, rule, _, _ in trace))
        if (
            nf == right
            and left != right
            and len(trace) >= int(config.min_steps)
            and len(used) >= int(config.min_core_size)
        ):
            return ScratchInstance(rules, left, right, used, trace)
    raise RuntimeError("failed to sample a positive scratch instance")


def _mutated_term(term, config):
    positions = list(_positions(term))
    random.shuffle(positions)
    for pos in positions:
        sub = _get_at(term, pos)
        if _is_app(sub):
            funcs = _unary_functions(config) if len(sub) == 2 else _binary_functions(config)
            choices = [f for f in funcs if f != sub[0]]
            if choices:
                return _set_at(term, pos, (random.choice(choices), *_args(sub)))
        elif not _is_var(sub):
            choices = [c for c in _constants(config) if c != sub]
            if choices:
                return _set_at(term, pos, random.choice(choices))
    return _sample_ground_term(config, max(0, _depth(term)))


def _sample_negative_rhs(left, positive_rhs, rules, config):
    nf_left, _ = _normalize(left, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
    for _ in range(int(config.max_tries)):
        if config.surface_match_negatives and random.random() < 0.7:
            candidate = _mutated_term(positive_rhs, config)
        else:
            candidate = _sample_ground_term(config, max(0, _depth(positive_rhs) + random.choice([-1, 0, 1])))
        nf_bad, bad_trace = _normalize(
            candidate, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size)
        )
        if nf_bad == nf_left:
            continue
        if abs(_size(nf_bad) - _size(positive_rhs)) > int(config.rhs_size_tolerance):
            continue
        if abs(_depth(nf_bad) - _depth(positive_rhs)) > 2:
            continue
        if _rewrite_once(nf_bad, rules) is None:
            return nf_bad, bad_trace
    raise RuntimeError("failed to sample a scratch negative")


def _minimize_core(lhs, rhs, used, rules, config):
    core = set(used)
    changed = True
    while changed:
        changed = False
        for name in sorted(core):
            candidate = core - {name}
            if _certify_equivalent(lhs, rhs, _ordered(candidate, rules), config):
                core = candidate
                changed = True
                break
    return tuple(name for name in [rule.name for rule in rules] if name in core)


def _surface_overlap(rule, lhs, rhs):
    goal_symbols = _symbols(lhs) | _symbols(rhs)
    return len((_symbols(rule.lhs) | _symbols(rule.rhs)) & goal_symbols)


def _wrong_options(core, rules, lhs, rhs, config):
    core = tuple(core)
    all_names = [rule.name for rule in rules]
    outside = [name for name in all_names if name not in core]
    by_name = _rule_map(rules)
    wrong = []

    plausible = sorted(outside, key=lambda n: _surface_overlap(by_name[n], lhs, rhs), reverse=True)
    attempts = 0
    while len(wrong) < int(config.n_options) - 1 and attempts < int(config.max_tries) * 5:
        attempts += 1
        if not outside:
            break
        miss = random.choice(core)
        add = random.choice(plausible[: max(1, min(len(plausible), 4))] if random.random() < 0.5 else outside)
        candidate = tuple(name for name in all_names if name in ((set(core) - {miss}) | {add}))
        if candidate == core or candidate in wrong or len(candidate) != len(core):
            continue
        if _certify_inequivalent(lhs, rhs, _ordered(candidate, rules), config):
            wrong.append(candidate)
    if len(wrong) < int(config.n_options) - 1:
        for names in _same_size_subsets(all_names, len(core), set(core)):
            if len(wrong) >= int(config.n_options) - 1:
                break
            if names not in wrong and _certify_inequivalent(lhs, rhs, _ordered(names, rules), config):
                wrong.append(names)
    return wrong[: int(config.n_options) - 1]


def _same_size_subsets(names, size, forbidden):
    out = []
    pool = list(names)

    def rec(i, chosen):
        if len(out) >= 256:
            return
        if len(chosen) == size:
            cand = tuple(n for n in names if n in chosen)
            if set(cand) != forbidden:
                out.append(cand)
            return
        if i >= len(pool):
            return
        rec(i + 1, chosen | {pool[i]})
        rec(i + 1, chosen)

    rec(0, set())
    random.shuffle(out)
    return out


def _rule_text(rules, bullet=False):
    prefix = "- " if bullet else ""
    return "\n".join(f"{prefix}{rule.name}: {_render(rule.lhs)} -> {_render(rule.rhs)}" for rule in rules)


def _abc(i):
    return chr(ord("A") + i)


def _trace_data(trace):
    return [
        {"before": _render(before), "rule": rule_name, "position": list(pos), "after": _render(after)}
        for before, rule_name, pos, after in trace
    ]


def _trace_stats(trace):
    if not trace:
        return edict(fraction_nonroot_steps=0.0, mean_redex_depth=0.0, max_redex_depth=0)
    depths = [len(pos) for _, _, pos, _ in trace]
    return edict(
        fraction_nonroot_steps=sum(d > 0 for d in depths) / len(depths),
        mean_redex_depth=sum(depths) / len(depths),
        max_redex_depth=max(depths),
    )


def _rule_diagnostics(rules, config, core=None):
    samples = [_sample_ground_term(config, int(config.max_term_depth)) for _ in range(24)]
    fired = Counter()
    full_nfs = []
    for term in samples:
        nf, trace = _normalize(term, rules, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))
        full_nfs.append(nf)
        fired.update(rule for _, rule, _, _ in trace)
    shadowed = 0
    for i, rule in enumerate(rules):
        if any(_match(earlier.lhs, rule.lhs) is not None for earlier in rules[:i]):
            shadowed += 1
    redundant = 0
    for rule in rules:
        without = [r for r in rules if r.name != rule.name]
        if all(
            _normalize(term, without, int(config.max_norm_steps), max_size=int(config.max_intermediate_size))[0] == nf
            for term, nf in zip(samples, full_nfs)
        ):
            redundant += 1
    core_set = set(core or ())
    return edict(
        fraction_shadowed_rules=shadowed / len(rules) if rules else 0.0,
        fraction_rules_never_fired=sum(rule.name not in fired for rule in rules) / len(rules) if rules else 0.0,
        rule_redundancy_rate=redundant / len(rules) if rules else 0.0,
        fraction_rules_never_in_core=(
            sum(rule.name not in core_set for rule in rules) / len(rules) if core is not None and rules else None
        ),
    )


def _rule_data(rule):
    return {
        "name": rule.name,
        "lhs": _term_data(rule.lhs),
        "rhs": _term_data(rule.rhs),
        "mode": rule.mode,
        "precedence": list(rule.precedence),
    }


def _world_id(rules):
    text = "\n".join(f"{rule.name}:{_render(rule.lhs)}->{_render(rule.rhs)}" for rule in rules)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _build_world(config):
    rules = _sample_world(config)
    return ScratchWorld(rules=rules, world_id=_world_id(rules), diagnostics=_rule_diagnostics(rules, config))


def _next_world(task):
    if getattr(task, "_world_uses_left", 0) <= 0 or getattr(task, "_world_cache", None) is None:
        task._world_cache = _build_world(task.config)
        task._world_uses_left = max(1, int(task.config.world_cache_examples))
    task._world_uses_left -= 1
    return task._world_cache


def _extract_bool(answer):
    matches = re.findall(r"\bANSWER\s*[:\-]\s*(TRUE|FALSE)\b", str(answer).upper())
    if matches:
        return matches[-1].title()
    matches = re.findall(r"\b(TRUE|FALSE)\b", str(answer).upper())
    return matches[-1].title() if matches else None


def _extract_choice(answer):
    matches = re.findall(r"\bANSWER\s*[:\-]\s*([A-D])\b", str(answer).upper())
    if matches:
        return matches[-1]
    matches = re.findall(r"\b([A-D])\b", str(answer).upper())
    return matches[-1] if matches else None


def _extract_free_answer(answer):
    matches = re.findall(r"\bANSWER\s*[:\-]\s*([^\n\r]+)", str(answer), flags=re.IGNORECASE)
    return matches[-1].strip() if matches else str(answer).strip()


def _parse_rendered_term(text):
    tokens = re.findall(r"[A-Za-z]\w*|\(|\)|,", str(text))
    if not tokens:
        raise ValueError("empty term")
    i = 0

    def parse():
        nonlocal i
        if i >= len(tokens) or not re.fullmatch(r"[A-Za-z]\w*", tokens[i]):
            raise ValueError("expected symbol")
        symbol = tokens[i]
        i += 1
        if i < len(tokens) and tokens[i] == "(":
            i += 1
            args = []
            if i < len(tokens) and tokens[i] != ")":
                while True:
                    args.append(parse())
                    if i < len(tokens) and tokens[i] == ",":
                        i += 1
                        continue
                    break
            if i >= len(tokens) or tokens[i] != ")":
                raise ValueError("expected closing parenthesis")
            i += 1
            return (symbol, *args)
        return symbol

    term = parse()
    if i != len(tokens):
        raise ValueError("trailing tokens")
    return term


class MathScratchEntailment(DevTask):
    """True/False equivalence by normal form in a terminating rewrite world."""

    summary = "Decide expression equivalence by normalization across generated terminating symbolic rewrite systems."

    config_cls = ScratchConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or ScratchConfig(), **kwargs)
        self._world_cache = None
        self._world_uses_left = 0

    def generate_entry(self):
        for _ in range(int(self.config.max_tries)):
            try:
                world = _next_world(self)
                rules = world.rules
                inst = _sample_positive_instance(rules, self.config)
                positive = random.random() < 0.5
                rhs = inst.rhs
                negative_trace = []
                if not positive:
                    rhs, negative_trace = _sample_negative_rhs(inst.lhs, inst.rhs, rules, self.config)
                normal_left, left_trace = _normalize(
                    inst.lhs, rules, int(self.config.max_norm_steps), max_size=int(self.config.max_intermediate_size)
                )
                normal_right, right_trace = _normalize(
                    rhs, rules, int(self.config.max_norm_steps), max_size=int(self.config.max_intermediate_size)
                )
                trace_stats = _trace_stats(left_trace)
                meta = edict(
                    rules=[rule.name for rule in rules],
                    world_id=world.world_id,
                    raw_rules=[_rule_data(rule) for rule in rules],
                    rule_mode=self.config.rule_mode,
                    left=_render(inst.lhs),
                    right=_render(rhs),
                    raw_left=_term_data(inst.lhs),
                    raw_right=_term_data(rhs),
                    normal_left=_render(normal_left),
                    normal_right=_render(normal_right),
                    positive=positive,
                    proof=_trace_data(inst.proof),
                    left_trace=_trace_data(left_trace),
                    right_trace=_trace_data(right_trace),
                    negative_candidate_trace=_trace_data(negative_trace),
                    used=inst.used,
                    positive_rhs_size=_size(inst.rhs),
                    displayed_rhs_size=_size(rhs),
                    rhs_size_delta=_size(rhs) - _size(inst.rhs),
                    diagnostics=edict(trace=trace_stats, world=world.diagnostics),
                    source="scratch",
                )
                meta.payload = {
                    "rules": _rule_text(rules),
                    "left": meta.left,
                    "right": meta.right,
                }
                return Entry(meta, "True" if positive else "False")
            except RuntimeError:
                self._world_uses_left = 0
                continue
        raise RuntimeError("failed to generate MathScratchEntailment")

    def render_prompt(self, metadata):
        return (
            "Using only the rules below, do Left and Right reduce to the same final expression?\n\n"
            "Rules may be applied repeatedly to any matching subexpression. "
            "Variables like X and Y may match compound expressions.\n\n"
            "End with: Answer: True or Answer: False.\n\n"
            f"{render_payload(metadata.payload)}"
        )

    def score_answer(self, answer, entry):
        return float(_extract_bool(answer) == entry.answer)


class MathScratchCoreSelect(DevTask):
    """Select a sufficient displayed rule subset for a scratch rewrite proof."""

    summary = "Select a sufficient displayed rule subset that proves equivalence in generated terminating rewrite systems."

    config_cls = ScratchConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or ScratchConfig(), **kwargs)
        self._world_cache = None
        self._world_uses_left = 0

    def generate_entry(self):
        for _ in range(int(self.config.max_tries)):
            try:
                world = _next_world(self)
                rules = world.rules
                inst = _sample_positive_instance(rules, self.config)
                core = _minimize_core(inst.lhs, inst.rhs, inst.used, rules, self.config)
                core_rules = _ordered(core, rules)
                if len(core) < int(self.config.min_core_size) or len(core) > int(self.config.max_core_size):
                    continue
                if not _certify_equivalent(inst.lhs, inst.rhs, core_rules, self.config):
                    continue
                wrong = _wrong_options(core, rules, inst.lhs, inst.rhs, self.config)
                if len(wrong) < int(self.config.n_options) - 1:
                    continue
                options = [core, *wrong[: int(self.config.n_options) - 1]]
                if len(set(options)) != int(self.config.n_options):
                    continue
                random.shuffle(options)
                answer = _abc(options.index(core))
                option_text = "\n".join(f"{_abc(i)}. [{', '.join(option)}]" for i, option in enumerate(options))
                normal_left, left_trace = _normalize(
                    inst.lhs, rules, int(self.config.max_norm_steps), max_size=int(self.config.max_intermediate_size)
                )
                normal_right, right_trace = _normalize(
                    inst.rhs, rules, int(self.config.max_norm_steps), max_size=int(self.config.max_intermediate_size)
                )
                trace_stats = _trace_stats(left_trace)
                core_diagnostics = edict(
                    fraction_rules_never_in_core=sum(rule.name not in set(core) for rule in rules) / len(rules)
                )
                meta = edict(
                    rules=[rule.name for rule in rules],
                    world_id=world.world_id,
                    raw_rules=[_rule_data(rule) for rule in rules],
                    rule_mode=self.config.rule_mode,
                    left=_render(inst.lhs),
                    right=_render(inst.rhs),
                    raw_left=_term_data(inst.lhs),
                    raw_right=_term_data(inst.rhs),
                    normal_left=_render(normal_left),
                    normal_right=_render(normal_right),
                    proof=_trace_data(inst.proof),
                    left_trace=_trace_data(left_trace),
                    right_trace=_trace_data(right_trace),
                    used=inst.used,
                    positive_rhs_size=_size(inst.rhs),
                    displayed_rhs_size=_size(inst.rhs),
                    rhs_size_delta=0,
                    diagnostics=edict(trace=trace_stats, world=world.diagnostics, core=core_diagnostics),
                    options=[list(option) for option in options],
                    raw_options=[list(option) for option in options],
                    core=list(core),
                    source="scratch",
                )
                meta.payload = {
                    "rule_catalog": _rule_text(rules, bullet=True),
                    "left": meta.left,
                    "right": meta.right,
                    "options": option_text,
                }
                return Entry(meta, answer)
            except RuntimeError:
                self._world_uses_left = 0
                continue
        raise RuntimeError("failed to generate MathScratchCoreSelect")

    def render_prompt(self, metadata):
        return (
            "Which option is sufficient to make Left and Right reduce to the same final expression?\n\n"
            "Use only the rules in that option. Rules may be applied repeatedly to any matching subexpression. "
            "Variables like X and Y may match compound expressions.\n\n"
            "End with: Answer: X, where X is A, B, C, or D.\n\n"
            f"{render_payload(metadata.payload)}"
        )

    def score_answer(self, answer, entry):
        return float(_extract_choice(answer) == entry.answer)


class MathScratchNormalize(DevTask):
    """Predict the normal form of one expression under the displayed rewrite rules."""

    summary = "Predict canonical normal forms of expressions under generated terminating symbolic rewrite systems."

    config_cls = ScratchConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config or ScratchConfig(), **kwargs)
        self._world_cache = None
        self._world_uses_left = 0

    def generate_entry(self):
        for _ in range(int(self.config.max_tries)):
            try:
                world = _next_world(self)
                rules = world.rules
                inst = _sample_positive_instance(rules, self.config)
                normal_left, left_trace = _normalize(
                    inst.lhs, rules, int(self.config.max_norm_steps), max_size=int(self.config.max_intermediate_size)
                )
                trace_stats = _trace_stats(left_trace)
                meta = edict(
                    rules=[rule.name for rule in rules],
                    world_id=world.world_id,
                    raw_rules=[_rule_data(rule) for rule in rules],
                    rule_mode=self.config.rule_mode,
                    left=_render(inst.lhs),
                    raw_left=_term_data(inst.lhs),
                    normal_form=_render(normal_left),
                    raw_normal_form=_term_data(normal_left),
                    proof=_trace_data(inst.proof),
                    left_trace=_trace_data(left_trace),
                    used=inst.used,
                    diagnostics=edict(trace=trace_stats, world=world.diagnostics),
                    source="scratch",
                )
                meta.payload = {
                    "rules": _rule_text(rules),
                    "left": meta.left,
                }
                return Entry(meta, meta.normal_form)
            except RuntimeError:
                self._world_uses_left = 0
                continue
        raise RuntimeError("failed to generate MathScratchNormalize")

    def render_prompt(self, metadata):
        return (
            "Using only the rules below, what final expression does Left reduce to?\n\n"
            "Rules may be applied repeatedly to any matching subexpression. "
            "Variables like X and Y may match compound expressions.\n\n"
            "End with: Answer: <final expression>.\n\n"
            f"{render_payload(metadata.payload)}"
        )

    def score_answer(self, answer, entry):
        try:
            got = _parse_rendered_term(_extract_free_answer(answer))
            expected = _parse_rendered_term(entry.answer)
        except ValueError:
            return 0.0
        return float(got == expected)
