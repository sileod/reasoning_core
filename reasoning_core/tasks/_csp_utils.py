"""Finite-domain CSP IR, solving, metrics, selection, rendering, and families.

This private module keeps the public task wrapper small while retaining one auditable
semantic implementation.
"""

from __future__ import annotations


"""Typed, prompt-independent finite-domain CSP intermediate representation."""


from dataclasses import dataclass
from functools import lru_cache, reduce
from math import gcd
from typing import Any, Mapping, Sequence

import z3


@dataclass(frozen=True)
class Var:
    name: str
    domain: tuple[int, ...]
    sort: str = "int"

    def __init__(self, name: str, domain: Sequence[int], sort: str = "int"):
        values = tuple(sorted(set(domain)))
        if not values:
            raise ValueError("a variable domain cannot be empty")
        if sort not in {"int", "bool"}:
            raise ValueError(f"unsupported sort: {sort}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", values)
        object.__setattr__(self, "sort", sort)

    def canonical(self):
        return ("var", self.sort, self.name, self.domain)


class Formula:
    def to_z3(self, ctx: Mapping[str, z3.ArithRef]):
        raise NotImplementedError

    def canonical(self):
        raise NotImplementedError

    def variables(self) -> frozenset[Var]:
        raise NotImplementedError

    def complexity(self) -> int:
        return 1

    def render(self, renderer) -> str:
        return renderer.render(self)


def _vkey(v: Var):
    return v.canonical()


@dataclass(frozen=True)
class _VarValue(Formula):
    x: Var
    value: int

    def variables(self):
        return frozenset((self.x,))


@dataclass(frozen=True)
class Eq(_VarValue):
    def to_z3(self, ctx): return ctx[self.x.name] == self.value
    def canonical(self): return ("eq", _vkey(self.x), self.value)


@dataclass(frozen=True)
class Ne(_VarValue):
    def to_z3(self, ctx): return ctx[self.x.name] != self.value
    def canonical(self): return ("ne", _vkey(self.x), self.value)


@dataclass(frozen=True)
class In(Formula):
    x: Var
    values: tuple[int, ...]

    def __init__(self, x, values):
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "values", tuple(sorted(set(values))))

    def to_z3(self, ctx): return z3.Or(*[ctx[self.x.name] == v for v in self.values])
    def canonical(self): return ("in", _vkey(self.x), self.values)
    def variables(self): return frozenset((self.x,))


@dataclass(frozen=True)
class _VarVar(Formula):
    x: Var
    y: Var

    def variables(self): return frozenset((self.x, self.y))

    def _ordered(self):
        return tuple(sorted((_vkey(self.x), _vkey(self.y))))


@dataclass(frozen=True)
class EqVar(_VarVar):
    def to_z3(self, ctx): return ctx[self.x.name] == ctx[self.y.name]
    def canonical(self): return ("eqvar",) + self._ordered()


@dataclass(frozen=True)
class NeVar(_VarVar):
    def to_z3(self, ctx): return ctx[self.x.name] != ctx[self.y.name]
    def canonical(self): return ("nevar",) + self._ordered()


@dataclass(frozen=True)
class Lt(_VarVar):
    def to_z3(self, ctx): return ctx[self.x.name] < ctx[self.y.name]
    def canonical(self): return ("lt", _vkey(self.x), _vkey(self.y))


@dataclass(frozen=True)
class Distance(Formula):
    x: Var
    y: Var
    k: int

    def to_z3(self, ctx): return z3.Abs(ctx[self.x.name] - ctx[self.y.name]) == self.k
    def canonical(self): return ("distance",) + tuple(sorted((_vkey(self.x), _vkey(self.y)))) + (self.k,)
    def variables(self): return frozenset((self.x, self.y))


@dataclass(frozen=True)
class Linear(Formula):
    coeffs: tuple[int, ...]
    vars: tuple[Var, ...]
    op: str
    rhs: int

    def __init__(self, coeffs, vars, op, rhs):
        if len(coeffs) != len(vars) or not coeffs:
            raise ValueError("Linear needs equally sized non-empty coeffs and vars")
        if op not in {"==", "!=", "<=", ">="}:
            raise ValueError(f"unsupported linear operator: {op}")
        combined = {}
        for coefficient, var in zip(coeffs, vars):
            combined[var] = combined.get(var, 0) + int(coefficient)
        terms = sorted(((v, c) for v, c in combined.items() if c), key=lambda p: _vkey(p[0]))
        if not terms:
            raise ValueError("Linear cannot have all-zero coefficients")
        cs, vs = tuple(c for _, c in terms), tuple(v for v, _ in terms)
        divisor = reduce(gcd, (abs(c) for c in cs), 0)
        if divisor > 1 and rhs % divisor == 0:
            cs, rhs = tuple(c // divisor for c in cs), rhs // divisor
        if cs[0] < 0:
            cs, rhs = tuple(-c for c in cs), -rhs
            op = {"<=": ">=", ">=": "<="}.get(op, op)
        object.__setattr__(self, "coeffs", cs)
        object.__setattr__(self, "vars", vs)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "rhs", int(rhs))

    def _expr(self, ctx): return z3.Sum(*[a * ctx[v.name] for a, v in zip(self.coeffs, self.vars)])
    def to_z3(self, ctx):
        e = self._expr(ctx)
        return {"==": e == self.rhs, "!=": e != self.rhs, "<=": e <= self.rhs, ">=": e >= self.rhs}[self.op]
    def canonical(self): return ("linear", tuple(zip(self.coeffs, map(_vkey, self.vars))), self.op, self.rhs)
    def variables(self): return frozenset(self.vars)


@dataclass(frozen=True)
class Mod(Formula):
    expr: Linear
    modulus: int
    remainder: int

    def __init__(self, expr, modulus, remainder):
        if modulus < 2:
            raise ValueError("modulus must be at least 2")
        object.__setattr__(self, "expr", expr)
        object.__setattr__(self, "modulus", int(modulus))
        object.__setattr__(self, "remainder", int(remainder) % modulus)

    def to_z3(self, ctx): return self.expr._expr(ctx) % self.modulus == self.remainder
    def canonical(self):
        terms = tuple(zip(self.expr.coeffs, map(_vkey, self.expr.vars)))
        return ("mod", terms, self.modulus, self.remainder)
    def variables(self): return self.expr.variables()
    def complexity(self): return 1 + self.expr.complexity()


@dataclass(frozen=True)
class AllDifferent(Formula):
    vars: tuple[Var, ...]

    def __init__(self, vars):
        values = tuple(vars)
        if len(set(values)) != len(values):
            raise ValueError("AllDifferent variables must be distinct")
        object.__setattr__(self, "vars", tuple(sorted(values, key=_vkey)))
    def to_z3(self, ctx): return z3.Distinct(*[ctx[v.name] for v in self.vars])
    def canonical(self): return ("alldifferent", tuple(map(_vkey, self.vars)))
    def variables(self): return frozenset(self.vars)


@dataclass(frozen=True)
class Not(Formula):
    formula: Formula
    def to_z3(self, ctx): return z3.Not(self.formula.to_z3(ctx))
    def canonical(self): return ("not", self.formula.canonical())
    def variables(self): return self.formula.variables()
    def complexity(self): return 1 + self.formula.complexity()


class _Many(Formula):
    tag = "many"

    def __init__(self, formulas):
        flat = []
        for formula in formulas:
            flat.extend(formula.formulas if isinstance(formula, type(self)) else (formula,))
        unique = {f.canonical(): f for f in flat}
        object.__setattr__(self, "formulas", tuple(unique[k] for k in sorted(unique, key=repr)))

    def canonical(self): return (self.tag, tuple(f.canonical() for f in self.formulas))
    def variables(self): return frozenset().union(*(f.variables() for f in self.formulas))
    def complexity(self): return 1 + sum(f.complexity() for f in self.formulas)
    def __eq__(self, other): return type(self) is type(other) and self.canonical() == other.canonical()
    def __hash__(self): return hash(self.canonical())


class Or(_Many):
    tag = "or"
    def to_z3(self, ctx): return z3.Or(*[f.to_z3(ctx) for f in self.formulas])


@dataclass(frozen=True)
class Xor(Formula):
    """Exactly one child holds; nesting and repeated children are significant."""

    formulas: tuple[Formula, ...]
    tag = "xor"

    def __init__(self, formulas):
        values = tuple(sorted(formulas, key=lambda f: repr(f.canonical())))
        object.__setattr__(self, "formulas", values)

    def to_z3(self, ctx):
        if not self.formulas: return z3.BoolVal(False)
        return z3.PbEq([(f.to_z3(ctx), 1) for f in self.formulas], 1)
    def canonical(self): return (self.tag, tuple(f.canonical() for f in self.formulas))
    def variables(self): return frozenset().union(*(f.variables() for f in self.formulas))
    def complexity(self): return 1 + sum(f.complexity() for f in self.formulas)


@dataclass(frozen=True)
class Implies(Formula):
    a: Formula
    b: Formula
    def to_z3(self, ctx): return z3.Implies(self.a.to_z3(ctx), self.b.to_z3(ctx))
    def canonical(self): return ("implies", self.a.canonical(), self.b.canonical())
    def variables(self): return self.a.variables() | self.b.variables()
    def complexity(self): return 1 + self.a.complexity() + self.b.complexity()


@dataclass(frozen=True)
class _Cardinality(Formula):
    k: int
    formulas: tuple[Formula, ...]
    tag = "cardinality"

    def __init__(self, k, formulas):
        fs = tuple(sorted(formulas, key=lambda f: repr(f.canonical())))
        object.__setattr__(self, "k", int(k)); object.__setattr__(self, "formulas", fs)
    def canonical(self): return (self.tag, self.k, tuple(f.canonical() for f in self.formulas))
    def variables(self): return frozenset().union(*(f.variables() for f in self.formulas))
    def complexity(self): return 1 + sum(f.complexity() for f in self.formulas)


class Exactly(_Cardinality):
    tag = "exactly"
    def to_z3(self, ctx): return z3.PbEq([(f.to_z3(ctx), 1) for f in self.formulas], self.k)


class AtMost(_Cardinality):
    tag = "atmost"
    def to_z3(self, ctx): return z3.PbLe([(f.to_z3(ctx), 1) for f in self.formulas], self.k)


def canonical_unique(formulas):
    """Deduplicate formulas by normalized semantic structure."""
    return list({f.canonical(): f for f in formulas}.values())


def operator_name(formula: Formula) -> str:
    return formula.canonical()[0]



"""One Z3-backed solver interface for every CSP family."""


from itertools import product

import z3



class CSPSolver:
    def __init__(self, variables, base=(), clues=()):
        self.variables = tuple(variables)
        if len({v.name for v in self.variables}) != len(self.variables):
            raise ValueError("variable names must be unique")
        self.base, self.clues = tuple(base), tuple(clues)
        self.ctx = {v.name: z3.Int(v.name) for v in self.variables}
        self._sat_cache = {}
        self._values_cache = {}
        self._solutions_cache = {}
        self._expression_cache = {}

    @staticmethod
    def _key(formulas):
        return tuple(sorted((f.canonical() for f in formulas), key=repr))

    def _expression(self, formula):
        key = formula.canonical()
        if key not in self._expression_cache:
            self._expression_cache[key] = formula.to_z3(self.ctx)
        return self._expression_cache[key]

    def solver(self, clues=None, extra=(), domains=None):
        solver = z3.Solver()
        for v in self.variables:
            values = v.domain if domains is None else domains[v]
            solver.add(z3.Or(*[self.ctx[v.name] == x for x in values]))
        for formula in (*self.base, *(self.clues if clues is None else clues), *extra):
            solver.add(self._expression(formula))
        return solver

    def is_sat(self, clues=None, extra=()):
        active = self.clues if clues is None else clues
        key = (self._key(active), self._key(extra))
        if key not in self._sat_cache:
            self._sat_cache[key] = self.solver(active, extra).check() == z3.sat
        return self._sat_cache[key]

    def possible_values(self, var, clues=None, extra=()):
        active = self.clues if clues is None else clues
        key = (var, self._key(active), self._key(extra))
        if key in self._values_cache: return list(self._values_cache[key])
        solver = self.solver(active, extra)
        out = [
            value for value in var.domain
            if solver.check(self.ctx[var.name] == value) == z3.sat
        ]
        self._values_cache[key] = tuple(out)
        return out

    def unique_value(self, var, clues=None):
        values = self.possible_values(var, clues)
        return values[0] if len(values) == 1 else None

    def solutions(self, clues=None, limit=None):
        active = self.clues if clues is None else clues
        key = (self._key(active), limit)
        if key in self._solutions_cache:
            solutions, overflow = self._solutions_cache[key]
            return solutions, overflow
        solver, out = self.solver(active), []
        while solver.check() == z3.sat:
            model = solver.model()
            row = tuple(model.eval(self.ctx[v.name], model_completion=True).as_long() for v in self.variables)
            out.append(row)
            if limit is not None and len(out) > limit:
                self._solutions_cache[key] = (None, True)
                return None, True
            solver.add(z3.Or(*[self.ctx[v.name] != x for v, x in zip(self.variables, row)]))
        result = (sorted(out), False)
        self._solutions_cache[key] = result
        return result

    def lex_solution(self, clues=None):
        optimizer = z3.Optimize(); optimizer.set(priority="lex")
        for assertion in self.solver(clues).assertions(): optimizer.add(assertion)
        for var in self.variables: optimizer.minimize(self.ctx[var.name])
        if optimizer.check() != z3.sat: return None
        model = optimizer.model()
        return tuple(model.eval(self.ctx[v.name], model_completion=True).as_long() for v in self.variables)

    def entails(self, formula, clues=None):
        return not self.is_sat(clues, (Not(formula),))

    def refutation_core(self, var, wrong_value, clues=None):
        active = list(self.clues if clues is None else clues)
        if self.is_sat(active, (Eq(var, wrong_value),)): return []
        return self._sampled_core(active, (Eq(var, wrong_value),))

    def _sampled_core(self, active, extra):
        canonical = sorted(active, key=lambda c: repr(c.canonical()))
        orders = [canonical, list(reversed(canonical))]
        cores = []
        for order in orders:
            candidate = list(active)
            for clue in order:
                trial = candidate.copy(); trial.remove(clue)
                if not self.is_sat(trial, extra): candidate = trial
            cores.append(candidate)
        return min(cores, key=len)

    def possible_truth_values(self, formula, clues=None):
        active = self.clues if clues is None else clues
        return [value for value, condition in ((True, formula), (False, Not(formula)))
                if self.is_sat(active, (condition,))]

    def formula_refutation_core(self, forbidden_formula, clues=None):
        active = list(self.clues if clues is None else clues)
        if self.is_sat(active, (forbidden_formula,)): return []
        return self._sampled_core(active, (forbidden_formula,))

    def full_unique(self, clues=None):
        solutions, overflow = self.solutions(clues, limit=1)
        return not overflow and len(solutions) == 1



"""Concise deterministic renderers, separate from CSP semantics."""



class SymbolicRenderer:
    def name(self, var): return var.name

    def render_embedded(self, formula):
        return self.render(formula).rstrip(".")

    def render_many(self, formulas, separator):
        return separator.join(f"({self.render_embedded(formula)})" for formula in formulas)

    def render(self, f):
        n = self.name
        if isinstance(f, Eq): return f"{n(f.x)} = {f.value}"
        if isinstance(f, Ne): return f"{n(f.x)} != {f.value}"
        if isinstance(f, EqVar): return f"{n(f.x)} = {n(f.y)}"
        if isinstance(f, NeVar): return f"{n(f.x)} != {n(f.y)}"
        if isinstance(f, Lt): return f"{n(f.x)} < {n(f.y)}"
        if isinstance(f, Distance): return f"|{n(f.x)} - {n(f.y)}| = {f.k}"
        if isinstance(f, In): return f"{n(f.x)} in {{{', '.join(map(str, f.values))}}}"
        if isinstance(f, AllDifferent): return f"AllDifferent({', '.join(n(v) for v in f.vars)})"
        if isinstance(f, Linear):
            terms = [n(v) if a == 1 else f"-{n(v)}" if a == -1 else f"{a}*{n(v)}" for a, v in zip(f.coeffs, f.vars)]
            return f"{' + '.join(terms).replace('+ -', '- ')} {f.op} {f.rhs}"
        if isinstance(f, Mod): return f"({self.render(f.expr).rsplit(' ', 2)[0]}) % {f.modulus} = {f.remainder}"
        if isinstance(f, Not): return f"not ({self.render(f.formula)})"
        if isinstance(f, Implies): return f"if {self.render_embedded(f.a)}, then {self.render_embedded(f.b)}"
        if isinstance(f, (Or, Xor)):
            return self.render_many(f.formulas, " or " if isinstance(f, Or) else " xor ")
        if isinstance(f, (Exactly, AtMost)):
            label = "Exactly" if isinstance(f, Exactly) else "At most"
            return f"{label} {f.k} of [{'; '.join(self.render_embedded(x) for x in f.formulas)}]"
        raise TypeError(type(f).__name__)


class AssignmentRenderer(SymbolicRenderer):
    def __init__(self, labels, people): self.labels, self.people = labels, people
    def name(self, var): return self.labels[var.name]
    def role(self, var):
        category, value = var.name.split("_", 1)
        return {"pet": f"{value} owner", "drink": f"{value} drinker",
                "hobby": f"{value} player"}.get(category, self.name(var))
    def render(self, f):
        if isinstance(f, Eq): return f"The {self.role(f.x)} is {self.people[f.value]}."
        if isinstance(f, Ne): return f"The {self.role(f.x)} is not {self.people[f.value]}."
        if isinstance(f, EqVar): return f"The {self.role(f.x)} is the {self.role(f.y)}."
        if isinstance(f, NeVar): return f"The {self.role(f.x)} and {self.role(f.y)} are different people."
        if isinstance(f, Lt): return f"The {self.role(f.x)} is before the {self.role(f.y)}."
        if isinstance(f, Distance):
            relation = "adjacent to" if f.k == 1 else f"{f.k} positions from"
            return f"The {self.role(f.x)} is {relation} the {self.role(f.y)}."
        return super().render(f)



"""Solver-independent, interpretable finite-domain difficulty measurements."""


from collections import Counter
from itertools import product
from statistics import median

import z3



@lru_cache(maxsize=100_000)
def _supported_cached(formula, target, value, domain_items):
    """Whether a value has local support in one constraint (generalized arc consistency)."""
    variables = sorted(formula.variables(), key=lambda v: v.name)
    domains = dict(domain_items)
    ctx = {v.name: z3.Int(v.name) for v in variables}
    expression = formula.to_z3(ctx)
    for values in product(*(domains[v] if v != target else (value,) for v in variables)):
        substituted = z3.substitute(expression, *[(ctx[v.name], z3.IntVal(x)) for v, x in zip(variables, values)])
        if z3.is_true(z3.simplify(substituted)):
            return True
    return False


def _supported(formula, target, value, domains):
    domain_items = tuple((v, tuple(domains[v])) for v in sorted(formula.variables(), key=_vkey))
    return _supported_cached(formula, target, value, domain_items)


def propagate(variables, formulas, max_rounds=32, initial_domains=None):
    """Apply local support and singleton propagation until a fixed point."""
    domains = {v: list(v.domain if initial_domains is None else initial_domains[v]) for v in variables}
    rounds, forced_at = 0, {}
    for round_no in range(1, max_rounds + 1):
        previous = {v: list(values) for v, values in domains.items()}
        proposed = {v: list(values) for v, values in domains.items()}
        for formula in formulas:
            if any(not previous[v] for v in formula.variables()): continue
            if isinstance(formula, AllDifferent):
                singletons = {previous[v][0] for v in formula.vars if len(previous[v]) == 1}
                for var in formula.vars:
                    if len(previous[var]) > 1:
                        proposed[var] = [x for x in proposed[var] if x not in singletons]
                continue
            # Large global constraints have specialized propagation above; avoid
            # exponential tuple enumeration for arbitrary formulas.
            if len(formula.variables()) > 5: continue
            for var in formula.variables():
                supported = [x for x in previous[var] if _supported(formula, var, x, previous)]
                proposed[var] = [x for x in proposed[var] if x in supported]
        changed = proposed != previous
        domains = proposed
        for var in variables:
            if len(domains[var]) == 1 and len(previous[var]) > 1:
                forced_at.setdefault(var.name, round_no)
        if not changed: break
        rounds = round_no
    return domains, rounds, forced_at


def backbone_saturation(solver, clues):
    zsolver = solver.solver(clues)
    if zsolver.check() != z3.sat: return 0.0
    model, forced = zsolver.model(), 0
    for var in solver.variables:
        value = model.eval(solver.ctx[var.name], model_completion=True)
        zsolver.push(); zsolver.add(solver.ctx[var.name] != value)
        forced += zsolver.check() == z3.unsat
        zsolver.pop()
    return forced / len(solver.variables)


def first_solution_search_metrics(solver, clues, domains):
    """Propagating fail-first search until the first solution; not a uniqueness proof."""
    guesses = failed = max_depth = 0

    def visit(current, depth):
        nonlocal guesses, failed, max_depth
        current, _, _ = propagate(solver.variables, (*solver.base, *clues), initial_domains=current)
        max_depth = max(max_depth, depth)
        if not solver.solver(clues, domains=current).check() == z3.sat:
            failed += 1; return False
        open_vars = [v for v in solver.variables if len(current[v]) > 1]
        if not open_vars: return True
        var = min(open_vars, key=lambda v: (len(current[v]), v.name)); guesses += 1
        for value in current[var]:
            child = {v: list(xs) for v, xs in current.items()}; child[var] = [value]
            if visit(child, depth + 1): return True
        return False

    visit(domains, 0)
    return {"guesses": guesses, "failed_branches": failed, "max_depth": max_depth}


def _groups_for(variables, group_of):
    groups = set()
    for var in variables:
        value = group_of.get(var, "generic") if group_of else "generic"
        groups.update(value if isinstance(value, (tuple, list, set)) else (value,))
    return sorted(groups)


def analyze(solver: CSPSolver, clues, query, group_of=None, base_full_essential=None):
    base_values = solver.possible_values(query.var, ())
    values = solver.possible_values(query.var, clues)
    saturation = backbone_saturation(solver, clues)
    leave_one_out = []
    for i in range(len(clues)):
        reduced = clues[:i] + clues[i + 1:]
        leave_one_out.append(round(saturation - backbone_saturation(solver, reduced), 4))
    cores = [solver.refutation_core(query.var, v, clues) for v in query.var.domain if v not in values]
    domains, rounds, forced_at = propagate(solver.variables, (*solver.base, *clues))
    kinds = Counter(operator_name(c) for c in clues)
    essential_query = [solver.possible_values(query.var, clues[:i] + clues[i + 1:]) != values for i in range(len(clues))]
    full_unique = solver.full_unique(clues)
    essential_full = [full_unique and not solver.full_unique(clues[:i] + clues[i + 1:]) for i in range(len(clues))]
    essential_wrong = [any(clue in core for core in cores) for clue in clues]
    accepted = [a or b or c for a,b,c in zip(essential_query,essential_full,essential_wrong)]
    base_essential = []
    computed_base_full = []
    for i, invariant in enumerate(solver.base):
        reduced = CSPSolver(solver.variables, solver.base[:i] + solver.base[i+1:])
        base_essential.append(reduced.possible_values(query.var, clues) != values)
        if base_full_essential is None:
            computed_base_full.append(full_unique and not reduced.full_unique(clues))
    if base_full_essential is None: base_full_essential = computed_base_full
    without_alldifferent = tuple(f for f in solver.base if not isinstance(f, AllDifferent))
    alldifferent_collectively_essential = False
    if len(without_alldifferent) != len(solver.base):
        reduced = CSPSolver(solver.variables, without_alldifferent)
        alldifferent_collectively_essential = (
            reduced.possible_values(query.var, clues) != values
            or (full_unique and not reduced.full_unique(clues))
        )
    groups = _groups_for({v for c in clues for v in c.variables()}, group_of)
    core_details = [{
        "size": len(core),
        "variables_touched": len(set().union(*(c.variables() for c in core))) if core else 0,
        "operator_types": sorted({operator_name(c) for c in core}),
        "semantic_groups": _groups_for({v for c in core for v in c.variables()}, group_of),
    } for core in cores]
    return {
        "backbone_saturation": round(saturation, 4),
        "leave_one_out_effect": leave_one_out,
        "query_domain_before": base_values,
        "query_domain_after": values,
        "wrong_answer_core_sizes": [len(c) for c in cores],
        "wrong_answer_cores": core_details,
        "sampled_min_wrong_answer_core_size": min(map(len, cores), default=0),
        # Compatibility alias; this is the smallest of two deterministic
        # inclusion-minimal cores, not a cardinality proof.
        "minimum_wrong_answer_core_size": min(map(len, cores), default=0),
        "median_wrong_answer_core_size": median(map(len, cores)) if cores else 0,
        "operator_histogram": dict(sorted(kinds.items())),
        "variables_touched": len(set().union(*(c.variables() for c in clues))),
        "semantic_groups_touched": groups,
        "propagation_rounds": rounds,
        "query_forced_round": forced_at.get(query.var.name),
        "essential_for_query": essential_query,
        "essential_for_full_solution": essential_full,
        "essential_for_some_wrong_answer": essential_wrong,
        "displayed_clue_essentiality": round(sum(accepted) / len(clues), 4) if clues else 0,
        "query_essentiality": round(sum(essential_query) / len(clues), 4) if clues else 0,
        "base_invariant_essential": base_essential,
        "base_invariant_essential_for_full_solution": base_full_essential,
        "alldifferent_invariant_essential": any(
            (query_flag or full_flag) and isinstance(formula, AllDifferent)
            for query_flag, full_flag, formula in zip(base_essential, base_full_essential, solver.base)
        ) or alldifferent_collectively_essential,
        "global_invariant_essential": any(
            (query_flag or full_flag) and len(formula.variables()) > 1
            for query_flag, full_flag, formula in zip(base_essential, base_full_essential, solver.base)
        ) or alldifferent_collectively_essential,
        "first_solution_search": first_solution_search_metrics(solver, clues, domains),
    }


def split_key(family, base, clues, query_type):
    """A name-invariant structural key suitable for leakage-resistant splits."""
    def skeleton(formula):
        if isinstance(formula, Eq): return ("eq", formula.value)
        if isinstance(formula, Ne): return ("ne", formula.value)
        if isinstance(formula, In): return ("in", formula.values)
        if isinstance(formula, (EqVar, NeVar, Lt)): return (operator_name(formula),)
        if isinstance(formula, Distance): return ("distance", formula.k)
        if isinstance(formula, Linear): return ("linear", formula.op, formula.rhs)
        if isinstance(formula, Mod): return ("mod", formula.modulus, formula.remainder)
        if isinstance(formula, AllDifferent): return ("alldifferent", len(formula.vars))
        if isinstance(formula, Not): return ("not", skeleton(formula.formula))
        if isinstance(formula, Implies):
            return ("implies", skeleton(formula.a), skeleton(formula.b))
        if isinstance(formula, (Or, Xor)):
            return (operator_name(formula), tuple(sorted(map(skeleton, formula.formulas), key=repr)))
        if isinstance(formula, (Exactly, AtMost)):
            return (operator_name(formula), formula.k,
                    tuple(sorted(map(skeleton, formula.formulas), key=repr)))
        raise TypeError(type(formula).__name__)

    def roles(formula, prefix=""):
        if isinstance(formula, (Eq, Ne, In)): return [(formula.x, prefix + "subject")]
        if isinstance(formula, Lt): return [(formula.x, prefix + "left"), (formula.y, prefix + "right")]
        if isinstance(formula, (EqVar, NeVar, Distance)):
            return [(v, prefix + "member") for v in sorted((formula.x, formula.y), key=_vkey)]
        if isinstance(formula, Linear):
            return [(v, f"{prefix}coefficient:{a}") for a, v in zip(formula.coeffs, formula.vars)]
        if isinstance(formula, Mod): return roles(formula.expr, prefix + "mod:")
        if isinstance(formula, AllDifferent): return [(v, prefix + "member") for v in formula.vars]
        if isinstance(formula, Not): return roles(formula.formula, prefix + "not:")
        if isinstance(formula, Implies):
            return roles(formula.a, prefix + "antecedent:") + roles(formula.b, prefix + "consequent:")
        if isinstance(formula, (Or, Xor, Exactly, AtMost)):
            return [
                item
                for child in formula.formulas
                for item in roles(
                    child, f"{prefix}{operator_name(formula)}:child:{skeleton(child)!r}:"
                )
            ]
        return [(v, prefix + "member") for v in sorted(formula.variables(), key=_vkey)]

    import networkx as nx
    graph = nx.Graph()
    formulas = list(base) + list(clues)
    variables = sorted(set().union(*(f.variables() for f in formulas)), key=lambda v: v.name)
    for i, var in enumerate(variables):
        graph.add_node(f"v{i}", label=f"var:{var.sort}:{var.domain!r}")
    indices = {v: i for i,v in enumerate(variables)}
    for i, formula in enumerate(formulas):
        source = "base" if i < len(base) else "clue"
        node = f"c{i}"; graph.add_node(node, label=f"{source}:{skeleton(formula)!r}")
        for j, (var, role) in enumerate(roles(formula)):
            role_node = f"r{i}:{j}"
            graph.add_node(role_node, label=f"role:{role}")
            graph.add_edge(node, role_node); graph.add_edge(role_node, f"v{indices[var]}")
    incidence_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")
    return {
        "family": family,
        "base_constraint_skeleton": sorted((repr(skeleton(x)) for x in base)),
        "clue_formula_skeletons": sorted((repr(skeleton(x)) for x in clues)),
        "query_type": query_type,
        "incidence_graph": incidence_hash,
        "operator_histogram": dict(Counter(operator_name(x) for x in clues)),
    }



"""Query-aware multi-order clue selection."""


from dataclasses import dataclass



@dataclass(frozen=True)
class Query:
    kind: str
    var: object
    answer: object
    text: str


@dataclass
class SelectedInstance:
    query: Query
    clues: list
    metrics: dict
    objective: str = "unique_full_solution"


class Objective:
    name = "objective"
    def holds(self, solver, clues): raise NotImplementedError


class UniqueFullSolution(Objective):
    name = "unique_full_solution"
    def holds(self, solver, clues): return solver.full_unique(clues)


@dataclass(frozen=True)
class UniqueValue(Objective):
    var: Var
    value: int
    name = "unique_value"
    def holds(self, solver, clues): return solver.unique_value(self.var, clues) == self.value


@dataclass(frozen=True)
class Entails(Objective):
    formula: Formula
    name = "entails"
    def holds(self, solver, clues): return solver.entails(self.formula, clues)


@dataclass(frozen=True)
class Allows(Objective):
    formula: Formula
    name = "allows"
    def holds(self, solver, clues): return solver.is_sat(clues, (self.formula,))


@dataclass(frozen=True)
class Forbids(Objective):
    formula: Formula
    name = "forbids"
    def holds(self, solver, clues): return not solver.is_sat(clues, (self.formula,))


@dataclass(frozen=True)
class ExactlyNSolutions(Objective):
    n: int
    name = "exactly_n_solutions"
    def holds(self, solver, clues):
        solutions, overflow = solver.solutions(clues, self.n)
        return not overflow and len(solutions) == self.n


class Consistent(Objective):
    name = "consistent"
    def holds(self, solver, clues): return solver.is_sat(clues)


class Inconsistent(Objective):
    name = "inconsistent"
    def holds(self, solver, clues): return not solver.is_sat(clues)


def _sufficient(solver, clues, query):
    if query.kind in {"scalar", "entity"}: return solver.unique_value(query.var, clues) == query.answer
    if query.kind == "count": return len(solver.possible_values(query.var, clues)) == query.answer
    if query.kind == "possibility":
        possible = query.answer[0]
        return (query.answer[1] in solver.possible_values(query.var, clues)) == possible
    return False


def _stratified_pool(pool, rng, per_operator=12, limit=48):
    groups = {}
    for clue in canonical_unique(pool): groups.setdefault(operator_name(clue), []).append(clue)
    selected = []
    for group in groups.values():
        rng.shuffle(group)
        non_unary = [c for c in group if len(c.variables()) > 1]
        unary = [c for c in group if len(c.variables()) == 1]
        selected.extend(non_unary[:per_operator] + unary[:min(6, per_operator)])
    rng.shuffle(selected)
    return selected[:limit]


def minimize_for_objective(solver, pool, objective, rng, n_orders=6):
    """Construct and delete clues while preserving an arbitrary semantic objective."""
    pool = _stratified_pool(pool, rng)
    if not objective.holds(solver, pool): return []
    candidates = []
    for _ in range(n_orders):
        order = list(pool); rng.shuffle(order)
        selected = []
        for start in range(0, len(order), 3):
            selected.extend(order[start:start + 3])
            if objective.holds(solver, selected): break
        if not objective.holds(solver, selected): continue
        # Try unary/simple clues first so relational and compositional clues survive
        # when either can carry the same global information.
        deletion = list(selected); rng.shuffle(deletion)
        deletion.sort(key=lambda c: (len(c.variables()) > 1, c.complexity()))
        for clue in deletion:
            trial = selected.copy(); trial.remove(clue)
            if objective.holds(solver, trial): selected = trial
        candidates.append(selected)
    return candidates


def minimize_system(solver, pool, rng, n_orders=6):
    return minimize_for_objective(solver, pool, UniqueFullSolution(), rng, n_orders)


def clue_query_effect(solver, clue, query):
    before = solver.possible_values(query.var, ())
    after = solver.possible_values(query.var, (clue,))
    return len(before) - len(after), after


def query_leakage_metrics(solver, clues, query):
    local = [clue for clue in clues if query.var in clue.variables()]
    effects = [clue_query_effect(solver, clue, query) for clue in local]
    return {
        "query_local_fraction": round(len(local) / len(clues), 4) if clues else 0,
        "query_local_unary_count": sum(len(clue.variables()) == 1 for clue in local),
        "maximum_single_clue_reduction": max((reduction for reduction, _ in effects), default=0),
        "single_clue_forces_query": any(len(after) == 1 for _, after in effects),
    }


def _query_leaks(solver, clues, query, difficulty, family=None):
    leakage = query_leakage_metrics(solver, clues, query)
    local_limit = (
        2 / 3 if family == "scheduling"
        else 0.5 if difficulty <= 0 or len(solver.variables) < 4
        else 0.4
    )
    if leakage["query_local_fraction"] > local_limit: return True
    if leakage["single_clue_forces_query"]: return True
    if difficulty >= 1 and leakage["query_local_unary_count"]: return True
    reduction_limit = 0.5 if difficulty <= 0 or family == "scheduling" else 0.34
    if leakage["maximum_single_clue_reduction"] / len(query.var.domain) > reduction_limit:
        return True
    return False


def _quality_ok(metrics, clues, query, family, difficulty):
    strict = difficulty >= 2
    medium_scale = metrics.get("total_variables", 0) >= {
        "grid": 16, "scheduling": 5, "graph": 5,
    }.get(family, 4)
    if metrics["displayed_clue_essentiality"] < 0.8: return False
    if len(metrics["operator_histogram"]) < 2: return False
    required_variables = 3 if family == "scheduling" else min(
        4, metrics.get("total_variables", 4)
    )
    if metrics["variables_touched"] < required_variables: return False
    core_floor = 2 if family == "scheduling" else 3 if strict and medium_scale else 2
    if metrics["sampled_min_wrong_answer_core_size"] < core_floor: return False
    if strict and medium_scale and metrics["query_forced_round"] == 1: return False
    essential = {
        operator_name(clue) for clue, matters in zip(clues, (
            a or b or c for a, b, c in zip(
                metrics["essential_for_query"], metrics["essential_for_full_solution"],
                metrics["essential_for_some_wrong_answer"],
            )
        )) if matters
    }
    if family == "assignment" and strict:
        if not metrics["global_invariant_essential"]: return False
        if not essential & {"ne", "nevar", "or", "xor", "exactly", "atmost"}: return False
        if not any(len(core["semantic_groups"]) >= 2 for core in metrics["wrong_answer_cores"]): return False
    if family == "grid" and strict:
        if not metrics["global_invariant_essential"] or not essential & {"lt", "distance"}: return False
        if not any(len(core["semantic_groups"]) >= 3 for core in metrics["wrong_answer_cores"]): return False
    if family == "scheduling" and strict:
        if not metrics["global_invariant_essential"]: return False
        if not essential & {"lt", "distance", "linear"}: return False
    if family == "numeric" and strict:
        if any(core["size"] and core["variables_touched"] < 2 for core in metrics["wrong_answer_cores"]): return False
        if len(metrics["operator_histogram"]) < 2: return False
    return True


def minimize(solver, pool, query, rng, n_orders=6):
    """Compatibility utility: globally minimize first, then evaluate one query."""
    systems = minimize_system(solver, pool, rng, n_orders)
    if not systems: return None
    clues = systems[0]
    return SelectedInstance(query, clues, analyze(solver, clues, query))


def select_instance(variables, base, pool, queries, rng, n_orders=6, family="numeric",
                    difficulty=0, group_of=None, target_constraints=None):
    """Mix globally unique systems with ambiguous systems having one forced value."""
    solver = CSPSolver(variables, base)
    preferred = {
        "assignment": {"nevar", "or", "xor", "exactly", "atmost", "lt", "distance"},
        "grid": {"lt", "distance"},
        "scheduling": {"lt", "distance", "linear"},
        "numeric": {"linear", "mod", "alldifferent"},
        "graph": {"in", "eqvar", "nevar", "or", "xor"},
        "sets": {"implies", "exactly", "atmost", "or", "xor"},
    }.get(family, set())
    query_order = list(queries); rng.shuffle(query_order)
    choices = []
    full_probability = {"scheduling": 0.2, "graph": 0.35}.get(family, 0.5)
    full_objective = rng.random() < full_probability
    work = []
    if full_objective:
        systems = minimize_system(solver, pool, rng, n_orders)
        work = [(query, clues, "unique_full_solution")
                for clues in systems for query in query_order]
    else:
        for query in query_order[:min(2, len(query_order))]:
            objective = UniqueValue(query.var, query.answer)
            systems = minimize_for_objective(solver, pool, objective, rng, n_orders)
            work.extend((query, clues, objective.name) for clues in systems
                        if not solver.full_unique(clues))
    work.sort(key=lambda item: (
        abs(len(item[1]) - target_constraints) if target_constraints else 0,
        -sum(operator_name(c) in preferred for c in item[1]),
        -len({operator_name(c) for c in item[1]}),
        sum(len(c.variables()) == 1 for c in item[1]),
        -sum(len(c.variables()) > 1 for c in item[1]),
        len(item[1]),
    ))
    seen = set()
    for query, clues, objective_name in work:
        key = (query.var, tuple(sorted(repr(c.canonical()) for c in clues)))
        if key in seen: continue
        seen.add(key)
        if len(seen) > 3: break
        if _query_leaks(solver, clues, query, difficulty, family): continue
        if len({operator_name(c) for c in clues}) < 2: continue
        required_variables = 3 if family == "scheduling" else min(4, len(variables))
        if len({v for c in clues for v in c.variables()}) < required_variables:
            continue
        values = solver.possible_values(query.var, clues)
        strict_scale = len(variables) >= 4
        core_floor = 3 if difficulty >= 2 and strict_scale and family != "scheduling" else 2
        if any(
            len(solver.refutation_core(query.var, value, clues)) < core_floor
            for value in query.var.domain if value not in values
        ): continue
        system_full_unique = solver.full_unique(clues)
        base_full_essential = [
            system_full_unique and not CSPSolver(
                variables, tuple(base[:i]) + tuple(base[i+1:])
            ).full_unique(clues)
            for i in range(len(base))
        ]
        metrics = analyze(
            solver, clues, query, group_of=group_of,
            base_full_essential=base_full_essential,
        )
        metrics.update(query_leakage_metrics(solver, clues, query))
        metrics.update({
            "total_variables": len(variables),
            "full_solution_unique": system_full_unique,
            "objective": objective_name,
        })
        if _quality_ok(metrics, clues, query, family, difficulty):
            choices.append(SelectedInstance(query, clues, metrics, objective_name))
    if not choices: return None
    return max(choices, key=lambda x: (
        -(abs(len(x.clues) - target_constraints) if target_constraints else 0),
        x.metrics["sampled_min_wrong_answer_core_size"],
        len(x.metrics["operator_histogram"]),
        x.metrics["query_forced_round"] or 0,
        -sum(len(c.variables()) == 1 for c in x.clues),
    ))



from dataclasses import dataclass, field

import z3



@dataclass
class World:
    variables: tuple
    witness: dict
    data: dict = field(default_factory=dict)


def holds(formula, witness):
    ctx = {v.name: z3.Int(v.name) for v in formula.variables()}
    expression = formula.to_z3(ctx)
    pairs = [(ctx[v.name], z3.IntVal(witness[v])) for v in formula.variables()]
    return z3.is_true(z3.simplify(z3.substitute(expression, *pairs)))


def true_composites(rng, true_atoms, false_atoms, limit=16):
    """Build shallow true formulas; false atoms are retained for counterfactual metadata."""
    out = []
    if not true_atoms or not false_atoms: return out
    for _ in range(limit):
        truth = rng.choice(true_atoms); miss = rng.choice(false_atoms)
        out.extend((Or((truth, miss)), Xor((truth, miss)), Implies(miss, rng.choice(true_atoms))))
    rng.shuffle(out)
    return out[:limit]


def false_composites(rng, true_atoms, false_atoms, limit=8):
    if not true_atoms or len(false_atoms) < 2: return []
    out = []
    for _ in range(limit):
        a,b = rng.sample(false_atoms, 2)
        out.extend((Or((a,b)), Implies(rng.choice(true_atoms), a)))
    rng.shuffle(out)
    return out[:limit]






class AssignmentFamily:
    name = "assignment"
    people_catalog = "Alice Bruno Clara David Elena Farah".split()
    catalogs = {
        "pet": "cat dog bird fish horse rabbit".split(),
        "drink": "tea milk juice water coffee cocoa".split(),
        "hobby": "chess music art dance tennis hiking".split(),
    }

    def sample_world(self, rng, size):
        n = max(3, min(6, size)); people = self.people_catalog[:n]
        variables, witness, labels, groups = [], {}, {}, {}
        for category, catalog in self.catalogs.items():
            group = []
            for value, owner in zip(catalog[:n], rng.sample(range(n), n)):
                var = Var(f"{category}_{value}", range(n), "int")
                variables.append(var); group.append(var); witness[var] = owner
                labels[var.name] = f"{value} ({category})"
            groups[category] = group
        group_of = {var: category for category, members in groups.items() for var in members}
        return World(tuple(variables), witness, {
            "people": people, "labels": labels, "groups": groups, "group_of": group_of,
        })

    def variables(self, world): return world.variables
    def base_constraints(self, world): return [AllDifferent(vs) for vs in world.data["groups"].values()]

    def candidate_atoms(self, world):
        true, false = [], []
        for var in world.variables:
            owner = world.witness[var]
            true += [Eq(var, owner), *[Ne(var, x) for x in var.domain if x != owner]]
            false += [Ne(var, owner), *[Eq(var, x) for x in var.domain if x != owner]]
        pairs = [(a, b) for i, a in enumerate(world.variables) for b in world.variables[i + 1:]
                 if a.name.split("_", 1)[0] != b.name.split("_", 1)[0]]
        for a, b in pairs:
            same = world.witness[a] == world.witness[b]
            true.append(EqVar(a, b) if same else NeVar(a, b))
            false.append(NeVar(a, b) if same else EqVar(a, b))
            if not same:
                earlier, later = (a, b) if world.witness[a] < world.witness[b] else (b, a)
                true.append(Lt(earlier, later)); false.append(Lt(later, earlier))
                distance = abs(world.witness[a] - world.witness[b])
                if distance <= 2: true.append(Distance(a, b, distance))
        return true, false

    def candidate_formulas(self, world, rng):
        true, false = self.candidate_atoms(world)
        return true + true_composites(rng, true, false, 18), false

    def queries(self, world):
        def question(var):
            category, value = var.name.split("_", 1)
            return {"pet": f"Who owns the {value}?", "drink": f"Who drinks {value}?",
                    "hobby": f"Who plays {value}?"}[category]
        return [Query("entity", v, world.witness[v], question(v)) for v in world.variables]

    def renderer(self, world): return AssignmentRenderer(world.data["labels"], world.data["people"])
    def answer(self, world, query): return world.data["people"][query.answer]
    def invariant(self, world): return "Each category is a one-to-one assignment."
    def domains_text(self, world):
        groups = world.data["groups"]
        lines = ["People left-to-right: " + ", ".join(world.data["people"]) + "."]
        lines += [
            f"{name.title()}: "
            + ", ".join(world.data["labels"][v.name].split(" (")[0] for v in variables)
            + "."
            for name, variables in groups.items()
        ]
        return "\n".join(lines)






class GraphFamily:
    name = "graph"
    edge_prob = .45
    def sample_world(self, rng, size):
        n, colors = max(4, min(8, size + 1)), 3
        variables = tuple(Var(f"v{i}", range(colors)) for i in range(n))
        witness = {v: rng.randrange(colors) for v in variables}
        edges = [(variables[i], variables[j]) for i in range(n) for j in range(i+1, n)
                 if witness[variables[i]] != witness[variables[j]] and rng.random() < self.edge_prob]
        return World(variables, witness, {
            "edges": edges, "colors": colors, "group_of": {v: "graph" for v in variables},
        })
    def variables(self, w): return w.variables
    def base_constraints(self, w): return [NeVar(a, b) for a, b in w.data["edges"]]
    def candidate_formulas(self, w, rng):
        true, false = [], []
        for v in w.variables:
            true += [Eq(v, w.witness[v]), *[Ne(v, x) for x in v.domain if x != w.witness[v]]]
            alternatives = [x for x in v.domain if x != w.witness[v]]
            if alternatives: true.append(In(v, (w.witness[v], rng.choice(alternatives))))
            false.append(Ne(v, w.witness[v]))
        for i, a in enumerate(w.variables):
            for b in w.variables[i+1:]:
                same = w.witness[a] == w.witness[b]
                true.append(EqVar(a,b) if same else NeVar(a,b))
                false.append(NeVar(a,b) if same else EqVar(a,b))
        return true + true_composites(rng, true, false, 10), false
    def queries(self, w): return [Query("scalar", v, w.witness[v], f"What color number is {v.name}?") for v in w.variables]
    def renderer(self, w): return SymbolicRenderer()
    def answer(self, w, q): return str(q.answer)
    def invariant(self, w): return "Adjacent vertices have different colors numbered 0..2."
    def domains_text(self, w): return "Edges: " + ", ".join(f"{a.name}-{b.name}" for a,b in w.data["edges"]) + "."






class GridFamily:
    name = "grid"
    def sample_world(self, rng, size):
        n = max(3, min(5, size)); symbols = rng.sample(range(1, n + 1), n)
        rows = [[symbols[(r + c) % n] for c in range(n)] for r in range(n)]
        rng.shuffle(rows)
        columns = rng.sample(range(n), n)
        rows = [[row[c] for c in columns] for row in rows]
        if rng.random() < 0.5: rows = [list(row) for row in zip(*rows)]
        variables = tuple(Var(f"r{r+1}c{c+1}", range(1, n + 1)) for r in range(n) for c in range(n))
        group_of = {v: (f"row{i//n+1}", f"column{i%n+1}") for i, v in enumerate(variables)}
        return World(variables, {v: rows[i // n][i % n] for i, v in enumerate(variables)}, {
            "n": n, "group_of": group_of,
        })
    def variables(self, w): return w.variables
    def base_constraints(self, w):
        n = w.data["n"]
        return [AllDifferent(w.variables[r*n:(r+1)*n]) for r in range(n)] + [AllDifferent(w.variables[c::n]) for c in range(n)]
    def candidate_formulas(self, w, rng):
        n, true, false = w.data["n"], [], []
        for v in w.variables:
            true.append(Eq(v, w.witness[v])); false.append(Ne(v, w.witness[v]))
            true.extend(Ne(v, x) for x in v.domain if x != w.witness[v])
        for r in range(n):
            for c in range(n):
                a = w.variables[r*n+c]
                for dr, dc in ((1, 0), (0, 1)):
                    if r+dr < n and c+dc < n:
                        b = w.variables[(r+dr)*n+c+dc]
                        (true if w.witness[a] < w.witness[b] else false).append(Lt(a, b))
                        (true if w.witness[b] < w.witness[a] else false).append(Lt(b, a))
        return true + true_composites(rng, true, false, 12), false
    def queries(self, w): return [Query("scalar", v, w.witness[v], f"What is {v.name}?") for v in w.variables]
    def renderer(self, w): return SymbolicRenderer()
    def answer(self, w, q): return str(q.answer)
    def invariant(self, w): return f"In this {w.data['n']}x{w.data['n']} grid, each row and column contains 1..{w.data['n']} once."
    def domains_text(self, w): return ""






class NumericFamily:
    name = "numeric"
    def __init__(self, coef_bound=3):
        self.coef_bound = coef_bound
        self.n_constraints = 3
        self.structure_mode = "any"
        self.max_arity = 3
        self.edge_prob = .25
        self.n_clusters = 3
        self.p_in = .6
        self.p_out = .1
        self.grid_width = None

    def _neighbors(self, rng, n, mode):
        neighbors = [set() for _ in range(n)]
        def link(i, j):
            if i != j: neighbors[i].add(j); neighbors[j].add(i)
        if mode in {"complete", "random"}:
            for i in range(n): neighbors[i] = set(range(n)) - {i}
        elif mode == "graph":
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < self.edge_prob: link(i, j)
        elif mode == "grid":
            width = self.grid_width or max(1, round(n ** .5))
            for i in range(n):
                row, column = divmod(i, width)
                for dr, dc in ((1, 0), (0, 1)):
                    j = (row + dr) * width + column + dc
                    if column + dc < width and j < n: link(i, j)
        elif mode == "clustered":
            cluster_count = max(1, min(self.n_clusters, n))
            clusters = [rng.randrange(cluster_count) for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    probability = self.p_in if clusters[i] == clusters[j] else self.p_out
                    if rng.random() < probability: link(i, j)
        for i in range(n):
            if not neighbors[i]: link(i, rng.choice([j for j in range(n) if j != i]))
        return tuple(tuple(sorted(items)) for items in neighbors)

    def sample_world(self, rng, size, max_domain=4):
        n = max(2, min(7, size))
        variables = tuple(Var(f"x{i}", range(rng.randint(2,max_domain)+1)) for i in range(n))
        mode = rng.choice(("random", "graph", "grid", "clustered")) if self.structure_mode == "any" else self.structure_mode
        return World(variables, {v: rng.choice(v.domain) for v in variables}, {
            "group_of": {v: "numeric" for v in variables},
            "structure_mode": mode, "neighbors": self._neighbors(rng, n, mode),
        })
    def variables(self, w): return w.variables
    def base_constraints(self, w): return []
    def candidate_formulas(self, w, rng):
        true, false = [], []
        for v in w.variables:
            true += [Eq(v,w.witness[v]), *[Ne(v,x) for x in v.domain if x != w.witness[v]]]
            false.append(Ne(v,w.witness[v]))
        for _ in range(max(5 * len(w.variables), 4 * self.n_constraints)):
            arity = rng.randint(1, min(self.max_arity, len(w.variables)))
            if arity == 1 or w.data["structure_mode"] == "random":
                scope = rng.sample(w.variables, arity)
            else:
                seed = rng.randrange(len(w.variables)); indices = {seed}
                frontier = set(w.data["neighbors"][seed])
                while len(indices) < arity and frontier:
                    item = rng.choice(tuple(frontier)); indices.add(item)
                    frontier.update(w.data["neighbors"][item]); frontier.difference_update(indices)
                if len(indices) < arity:
                    rest = [i for i in range(len(w.variables)) if i not in indices]
                    indices.update(rng.sample(rest, arity - len(indices)))
                scope = [w.variables[i] for i in sorted(indices)]
            coeffs = [rng.choice([x for x in range(-self.coef_bound,self.coef_bound+1) if x]) for _ in scope]
            value = sum(a*w.witness[v] for a,v in zip(coeffs,scope)); op = rng.choice(("==","<=",">=","!="))
            rhs = value if op == "==" else value+rng.randint(0,2) if op == "<=" else value-rng.randint(0,2) if op == ">=" else value+rng.choice((-2,-1,1,2))
            true.append(Linear(coeffs,scope,op,rhs))
            false_rhs = value+1 if op == "==" else value-1 if op == "<=" else value+1 if op == ">=" else value
            false.append(Linear(coeffs,scope,op,false_rhs))
            modulus = rng.randint(2,5); expr = Linear(coeffs,scope,"==",value)
            normalized_value = sum(a*w.witness[v] for a,v in zip(expr.coeffs,expr.vars))
            true.append(Mod(expr,modulus,normalized_value%modulus))
            false.append(Mod(expr,modulus,(normalized_value+1)%modulus))
        distinct = [v for v in w.variables if list(w.witness.values()).count(w.witness[v]) == 1]
        distinct = distinct[:self.max_arity]
        if len(distinct)>1: true.append(AllDifferent(distinct))
        equal_pairs = [(a,b) for i,a in enumerate(w.variables) for b in w.variables[i+1:]
                       if w.witness[a] == w.witness[b]]
        if equal_pairs: false.append(AllDifferent(rng.choice(equal_pairs)))
        return true + true_composites(rng, true, false, 12), false + false_composites(rng,true,false)
    def queries(self, w): return [Query("scalar", v, w.witness[v], f"What is {v.name}?") for v in w.variables]
    def renderer(self, w): return SymbolicRenderer()
    def answer(self, w, q): return str(q.answer)
    def invariant(self, w): return "All variables are integers in their listed finite domains."
    def domains_text(self, w): return "Domains: " + ", ".join(f"{v.name} in {{{', '.join(map(str,v.domain))}}}" for v in w.variables) + "."






class SchedulingFamily:
    name = "scheduling"
    def sample_world(self, rng, size):
        n = max(3, min(6, size)); names = [chr(65+i) for i in range(n)]
        durations = {name: rng.randint(1, 2) for name in names}
        order = rng.sample(names, n); starts, cursor = {}, 0
        for name in order:
            starts[name] = cursor; cursor += durations[name]
        horizon = cursor + rng.randint(0, max(1, n//2))
        variables = tuple(Var(name, range(horizon - durations[name] + 1)) for name in names)
        return World(variables, {v: starts[v.name] for v in variables}, {
            "n": n, "horizon": horizon, "durations": durations,
            "group_of": {v: "task" for v in variables},
        })
    def variables(self, w): return w.variables
    def base_constraints(self, w):
        durations = w.data["durations"]
        return [
            Or((
                Linear((1,-1),(a,b),"<=",-durations[a.name]),
                Linear((1,-1),(b,a),"<=",-durations[b.name]),
            ))
            for i,a in enumerate(w.variables) for b in w.variables[i+1:]
        ]
    def candidate_formulas(self, w, rng):
        true, false = [], []
        for v in w.variables:
            true.append(Eq(v, w.witness[v])); false.append(Ne(v, w.witness[v]))
            true.extend(Ne(v, x) for x in v.domain if x != w.witness[v])
        for i, a in enumerate(w.variables):
            for b in w.variables[i+1:]:
                earlier, later = (a,b) if w.witness[a] < w.witness[b] else (b,a)
                true.append(Lt(earlier, later)); false.append(Lt(later, earlier))
                if abs(w.witness[a]-w.witness[b]) <= 3: true.append(Distance(a,b,abs(w.witness[a]-w.witness[b])))
        return true + true_composites(rng, true, false, 12), false
    def queries(self, w): return [Query("scalar", v, w.witness[v], f"When does task {v.name} start?") for v in w.variables]
    def renderer(self, w): return SymbolicRenderer()
    def answer(self, w, q): return str(q.answer)
    def invariant(self, w):
        return "Integer starts; tasks finish by the horizon and share one non-overlapping resource."
    def domains_text(self, w):
        return "Horizon: 0..{}\nDurations: {}.".format(
            w.data["horizon"], ", ".join(f"{name}={duration}" for name,duration in w.data["durations"].items())
        )






class SetFamily:
    name = "sets"
    max_arity = 4
    n_constraints = 3
    def sample_world(self, rng, size):
        n = max(4, min(8, size + 2)); variables = tuple(Var(f"m{i}", (0,1), "bool") for i in range(n))
        witness = {v: rng.randrange(2) for v in variables}
        if not any(witness.values()): witness[rng.choice(variables)] = 1
        return World(variables, witness, {
            "n": n, "group_of": {v: "membership" for v in variables},
        })
    def variables(self, w): return w.variables
    def base_constraints(self, w): return []
    def candidate_formulas(self, w, rng):
        true, false = [], []
        for v in w.variables:
            true.append(Eq(v, w.witness[v])); false.append(Eq(v, 1-w.witness[v]))
        for _ in range(max(2 * w.data["n"], 2 * self.n_constraints) if self.max_arity >= 2 else 0):
            subset = tuple(rng.sample(w.variables, rng.randint(2, min(self.max_arity, len(w.variables)))))
            atoms = tuple(Eq(v, 1) for v in subset); count = sum(w.witness[v] for v in subset)
            true.extend((Exactly(count, atoms), AtMost(count, atoms)))
            if count < len(subset): false.append(Exactly(count+1, atoms))
        for _ in range(w.data["n"]):
            a, b = rng.sample(w.variables, 2)
            implication = Implies(Eq(a, 1), Eq(b, w.witness[b]))
            (true if holds(implication, w.witness) else false).append(implication)
        return true + true_composites(rng, true, false, 10), false
    def queries(self, w): return [Query("scalar", v, w.witness[v], f"Is {v.name} selected (1 or 0)?") for v in w.variables]
    def renderer(self, w): return SymbolicRenderer()
    def answer(self, w, q): return str(q.answer)
    def invariant(self, w): return "Each membership variable is 0 (not selected) or 1 (selected)."
    def domains_text(self, w): return ""




"""Thin family adapters over the common CSP semantic core."""


FAMILIES = {
    "assignment": AssignmentFamily(), "graph": GraphFamily(), "grid": GridFamily(),
    "numeric": NumericFamily(), "scheduling": SchedulingFamily(), "sets": SetFamily(),
}

ALIASES = {"attribute": "assignment", "linear": "numeric", "set": "sets"}


def get_family(name):
    return FAMILIES[ALIASES.get(name, name)]




"""Shared family-independent instance generation pipeline."""

from dataclasses import dataclass, replace



@dataclass
class GeneratedInstance:
    family: str
    world: object
    base: list
    clues: list
    counterfactuals: list
    counterfactual_pair: object
    unsat_clues: object
    sat_consistency_clues: object
    query: object
    answer: str
    metrics: dict
    split_key: dict
    renderer: object


def _minimal_unsat_core(solver, clues):
    active = list(clues)
    if solver.is_sat(active): return []
    for clue in list(active):
        trial = active.copy(); trial.remove(clue)
        if not solver.is_sat(trial): active = trial
    return active


def _has_complementary_pair(clues):
    eq = {(c.x, c.value) for c in clues if isinstance(c, Eq)}
    ne = {(c.x, c.value) for c in clues if isinstance(c, Ne)}
    return bool(eq & ne)


def semantic_unsat(solver, clues, false_pool, rng):
    candidates = [c for c in false_pool if len(c.variables()) >= 2]
    rng.shuffle(candidates)
    for mutation in candidates:
        core = _minimal_unsat_core(solver, [*clues, mutation])
        if (len(core) >= 3 and len(set().union(*(c.variables() for c in core))) >= 2
                and not _has_complementary_pair(core)):
            return core
    return None


def semantic_consistency_pair(solver, clues, true_pool, false_pool, rng):
    candidates = [c for c in false_pool if len(c.variables()) >= 2]
    rng.shuffle(candidates)
    for mutation in candidates:
        core = _minimal_unsat_core(solver, [*clues, mutation])
        if (len(core) < 3 or len(set().union(*(c.variables() for c in core))) < 2
                or _has_complementary_pair(core)):
            continue
        replacements = [c for c in true_pool if operator_name(c) == operator_name(mutation)
                        and c not in core and abs(c.complexity()-mutation.complexity()) <= 2]
        rng.shuffle(replacements)
        for replacement in replacements:
            sat_clues = [replacement if c == mutation else c for c in core]
            if not solver.is_sat(sat_clues): continue
            _, multiple = solver.solutions(sat_clues, limit=1)
            if multiple: return sat_clues, core
    return None, None


def graph_relation_candidate(world, base, clues, rng):
    """Find a same-color query whose answer requires several displayed clues."""
    solver = CSPSolver(world.variables, base)
    if solver.full_unique(clues): return None
    pairs = [(a,b) for i,a in enumerate(world.variables) for b in world.variables[i+1:]]
    rng.shuffle(pairs)
    for a,b in pairs:
        relation = EqVar(a,b)
        if len(solver.possible_truth_values(relation, ())) < 2: continue
        if any(len(solver.possible_truth_values(relation, (clue,))) == 1 for clue in clues): continue
        values = solver.possible_truth_values(relation, clues)
        if len(values) != 1: continue
        answer = values[0]
        forbidden = Not(relation) if answer else relation
        core = solver.formula_refutation_core(forbidden, clues)
        edge_essential = any(
            len(CSPSolver(world.variables, base[:i] + base[i+1:]).possible_truth_values(
                relation, clues
            )) > 1
            for i, edge in enumerate(base) if isinstance(edge, NeVar)
        )
        if len(core) >= 3 and edge_essential and any(len(c.variables()) > 1 for c in core):
            return a,b,answer,{
                "relation_core_size": len(core),
                "relation_core_operator_types": sorted({operator_name(c) for c in core}),
                "relation_single_clue_forces": False,
                "relation_expected": answer,
                "objective": "entails",
                "full_solution_unique": False,
                "graph_edge_essential": True,
            }
    return None


def _objective_metrics(solver, clues, schema, objective, essential, group_of=None):
    touched = {v for clue in clues for v in clue.variables()}
    return {
        "schema": schema,
        "objective": objective,
        "operator_histogram": dict(sorted(Counter(operator_name(c) for c in clues).items())),
        "variables_touched": len(touched),
        "semantic_groups_touched": _groups_for(touched, group_of),
        "essential_for_objective": essential,
        "displayed_clue_essentiality": round(sum(essential) / len(clues), 4) if clues else 0,
        "full_solution_unique": solver.full_unique(clues),
    }


def value_metrics(solver, clues, query, group_of=None, objective="unique_value"):
    metrics = analyze(solver, clues, query, group_of=group_of)
    essential = metrics["essential_for_query"]
    metrics.update(query_leakage_metrics(solver, clues, query))
    metrics.update({
        "schema": "value",
        "objective": objective,
        "essential_for_objective": essential,
        "displayed_clue_essentiality": round(sum(essential) / len(clues), 4) if clues else 0,
        "total_variables": len(solver.variables),
        "full_solution_unique": solver.full_unique(clues),
    })
    return metrics


def possibility_metrics(solver, clues, var, value, group_of=None):
    values = solver.possible_values(var, clues)
    possible = value in values
    core = [] if possible else solver.formula_refutation_core(Eq(var, value), clues)
    essential = [
        (value in solver.possible_values(var, clues[:i] + clues[i + 1:])) != possible
        for i in range(len(clues))
    ]
    metrics = _objective_metrics(
        solver, clues, "possibility", "allows" if possible else "forbids",
        essential, group_of,
    )
    metrics.update({
        "query_domain_before": solver.possible_values(var, ()),
        "query_domain_after": values,
        "possible_value_count": len(values),
        "possibility_expected": possible,
        "possibility_refutation_core_size": len(core),
        "possibility_core_operator_types": sorted({operator_name(c) for c in core}),
    })
    return metrics


def relation_metrics(solver, clues, relation, expected, group_of=None):
    before = solver.possible_truth_values(relation, ())
    after = solver.possible_truth_values(relation, clues)
    forbidden = Not(relation) if expected else relation
    core = solver.formula_refutation_core(forbidden, clues)
    essential = [
        solver.possible_truth_values(relation, clues[:i] + clues[i + 1:]) != after
        for i in range(len(clues))
    ]
    metrics = _objective_metrics(solver, clues, "relation", "entails", essential, group_of)
    metrics.update({
        "relation_truth_before": before,
        "relation_truth_after": after,
        "relation_expected": expected,
        "relation_core_size": len(core),
        "relation_core_operator_types": sorted({operator_name(c) for c in core}),
        "relation_single_clue_forces": any(
            len(solver.possible_truth_values(relation, (clue,))) == 1 for clue in clues
        ),
        "graph_edge_essential": any(
            len(CSPSolver(solver.variables, solver.base[:i] + solver.base[i + 1:]).possible_truth_values(
                relation, clues,
            )) > 1
            for i, edge in enumerate(solver.base) if isinstance(edge, NeVar)
        ),
    })
    return metrics


def consistency_metrics(solver, clues, group_of=None):
    consistent = solver.is_sat(clues)
    essential = [
        solver.is_sat(clues[:i] + clues[i + 1:]) != consistent
        for i in range(len(clues))
    ]
    metrics = _objective_metrics(
        solver, clues, "consistency", "consistent" if consistent else "inconsistent",
        essential, group_of,
    )
    _, multiple = solver.solutions(clues, limit=1)
    active = list(clues)
    if not consistent:
        for clue in list(active):
            trial = active.copy(); trial.remove(clue)
            if not solver.is_sat(trial): active = trial
    metrics.update({
        "is_consistent": consistent,
        "consistency_core_size": None if consistent else len(active),
        "multiple_full_solutions": multiple,
    })
    return metrics


def enumeration_metrics(solver, clues, mode, solutions=None, group_of=None):
    if mode in {"all", "all_solutions"}:
        if solutions is None: solutions, overflow = solver.solutions(clues)
        else: overflow = False
        if overflow: raise RuntimeError("Cannot measure an overflowing enumeration")
        outcome = solutions
        essential = []
        for i in range(len(clues)):
            reduced, _ = solver.solutions(clues[:i] + clues[i + 1:])
            essential.append(reduced != outcome)
        solution_count = len(outcome)
    else:
        outcome = solver.lex_solution(clues)
        essential = [
            solver.lex_solution(clues[:i] + clues[i + 1:]) != outcome
            for i in range(len(clues))
        ]
        solution_count = None
    metrics = _objective_metrics(solver, clues, "enumeration", mode, essential, group_of)
    metrics.update({
        "enumeration_mode": mode,
        "is_consistent": outcome is not None and outcome != [],
        "solution_count": solution_count,
    })
    return metrics


def generate_instance(family_name, rng, size, max_tries=64, n_orders=6, max_domain=4,
                      coef_bound=3, difficulty=0, require_consistency=False,
                      require_counterfactual=False, n_constraints=3, max_arity=3,
                      structure_mode="any", edge_prob=.25, n_clusters=3,
                      p_in=.6, p_out=.1, grid_width=None):
    family = get_family(family_name)
    if family.name == "numeric":
        family.coef_bound = coef_bound
        family.structure_mode, family.edge_prob = structure_mode, edge_prob
        family.n_clusters, family.p_in, family.p_out = n_clusters, p_in, p_out
        family.grid_width = grid_width
    if hasattr(family, "n_constraints"): family.n_constraints = n_constraints
    if hasattr(family, "max_arity"): family.max_arity = max_arity
    if family.name == "graph": family.edge_prob = edge_prob
    for _ in range(max_tries):
        world = family.sample_world(rng,size,max_domain) if family.name == "numeric" else family.sample_world(rng,size)
        base = family.base_constraints(world)
        pool, false_pool = family.candidate_formulas(world,rng)
        pool = [formula for formula in pool if len(formula.variables()) <= max_arity]
        false_pool = [formula for formula in false_pool if len(formula.variables()) <= max_arity]
        selected = select_instance(
            family.variables(world), base, pool, family.queries(world), rng, n_orders,
            family=family.name, difficulty=difficulty, group_of=world.data.get("group_of"),
            target_constraints=n_constraints,
        )
        if selected is None: continue
        solver = CSPSolver(family.variables(world), base)
        sat_consistency_clues = unsat_clues = None
        if require_consistency:
            sat_consistency_clues, unsat_clues = semantic_consistency_pair(
                solver, selected.clues, pool, false_pool, rng,
            )
            if unsat_clues is None: continue
        # Sample counterfactual mutations across operators instead of taking a
        # variable-order-biased prefix.
        by_operator = {}
        for formula in false_pool: by_operator.setdefault(operator_name(formula), []).append(formula)
        counterfactuals = [rng.choice(group) for group in by_operator.values() if group]
        remainder = [formula for group in by_operator.values() for formula in group
                     if formula not in counterfactuals]
        rng.shuffle(remainder); counterfactuals += remainder[:max(0, 12-len(counterfactuals))]
        counterfactual_pair = None
        if require_counterfactual:
            mutations = [(i, f) for i, clue in enumerate(selected.clues) for f in counterfactuals
                         if (operator_name(f) == operator_name(clue)
                             and abs(f.complexity() - clue.complexity()) <= 2)]
            rng.shuffle(mutations)
            for index, mutation in mutations[:16]:
                changed = selected.clues[:index] + [mutation] + selected.clues[index+1:]
                values = solver.possible_values(selected.query.var, changed)
                if len(values) == 1 and values[0] != selected.query.answer:
                    new_query = replace(selected.query, answer=values[0])
                    core_floor = (
                        2 if family.name == "scheduling"
                        else 3 if difficulty >= 2
                        else 2
                    )
                    cores = [solver.refutation_core(new_query.var, value, changed)
                             for value in new_query.var.domain if value != values[0]]
                    if any(len(core) < core_floor for core in cores): continue
                    if any(len(solver.possible_values(new_query.var, (clue,))) == 1 for clue in changed):
                        continue
                    if _query_leaks(solver, changed, new_query, difficulty, family.name):
                        continue
                    changed_metrics = analyze(
                        solver, changed, new_query, group_of=world.data.get("group_of"),
                    )
                    changed_metrics.update(query_leakage_metrics(solver, changed, new_query))
                    changed_metrics["total_variables"] = len(world.variables)
                    if not _quality_ok(changed_metrics, changed, new_query, family.name, difficulty):
                        continue
                    counterfactual_pair = (index, mutation, values[0], family.answer(world, new_query))
                    break
            if counterfactual_pair is None: continue
        return GeneratedInstance(
            family.name,world,base,selected.clues,counterfactuals,counterfactual_pair,
            unsat_clues,sat_consistency_clues,selected.query,
            family.answer(world,selected.query),selected.metrics,
            split_key(family.name,base,selected.clues,selected.query.kind),family.renderer(world),
        )
    raise RuntimeError(f"Failed to generate a quality {family.name} CSP after {max_tries} attempts")


def render_instance(instance):
    family = get_family(instance.family)
    world = instance.world
    header = "\n".join(filter(None, (family.domains_text(world), family.invariant(world))))
    clues = "\n".join(
        f"{i}. {clue.render(instance.renderer)}"
        for i, clue in enumerate(instance.clues, 1)
    )
    return (
        f"{header}\n\nConstraints:\n{clues}\n\n"
        f"Question: {instance.query.text}\n"
        "Answer with one name or integer."
    )
