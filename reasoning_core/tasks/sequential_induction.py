import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
from gramforge import generate, init_grammar

from reasoning_core.template import Config, Entry, Task, stochastic_rounding as sround


def Sequence_cfg(mode, recurrence_depth):
    """Grammar for the bounded polynomial recurrence DSL."""
    if mode != "simple":
        raise NotImplementedError(
            "Canonical SequentialInduction supports only n, U[n-i], integer "
            "constants, +, -, and *."
        )

    grammar = init_grammar(
        ["eq"],
        name=f"sequence_of_depth_{recurrence_depth}",
        preprocess_template=lambda value: value,
    )
    grammar("start(exp)", "{0}")
    grammar("exp", "n")
    grammar("exp(Ui)", "{0}")
    for lag in range(1, recurrence_depth + 1):
        grammar("Ui", f"U{lag}")
    grammar("exp(c_z)", "{0}")
    for value in range(-9, 10):
        grammar("c_z", str(value))
    grammar("exp(exp,bop,exp)", "({0} {1} {2})")
    for operator in "+-*":
        grammar("bop", operator)
    return grammar


# Sparse integer polynomials. Variable 0 is n; variable i is U[n-i].
def _poly_key(terms):
    return tuple(
        sorted(
            (powers, coefficient)
            for powers, coefficient in terms.items()
            if coefficient
        )
    )


def constant_poly(value, n_vars):
    return () if value == 0 else (((0,) * n_vars, int(value)),)


def variable_poly(index, n_vars):
    powers = [0] * n_vars
    powers[index] = 1
    return ((tuple(powers), 1),)


def add_poly(left, right):
    terms = dict(left)
    for powers, coefficient in right:
        terms[powers] = terms.get(powers, 0) + coefficient
    return _poly_key(terms)


def neg_poly(poly):
    return tuple((powers, -coefficient) for powers, coefficient in poly)


def sub_poly(left, right):
    return add_poly(left, neg_poly(right))


def mul_poly(left, right):
    if not left or not right:
        return ()
    terms = {}
    for left_powers, left_coefficient in left:
        for right_powers, right_coefficient in right:
            powers = tuple(a + b for a, b in zip(left_powers, right_powers))
            terms[powers] = terms.get(powers, 0) + left_coefficient * right_coefficient
    return _poly_key(terms)


def eval_poly(poly, values):
    total = 0
    for powers, coefficient in poly:
        term = coefficient
        for value, power in zip(values, powers):
            if power:
                term *= value**power
        total += term
    return total


def _sympify_formula(formula):
    n = sp.Symbol("n", integer=True, nonnegative=True)
    U = sp.IndexedBase("U")
    text = re.sub(r"(?<=\d)(?=n\b|U\s*\[)", "*", str(formula))
    return sp.sympify(text, locals={"n": n, "U": U})


def formula_degree(expr):
    expr = _sympify_formula(expr)
    n = next((symbol for symbol in expr.free_symbols if symbol.name == "n"), None)
    if n is None:
        n = sp.Symbol("n", integer=True, nonnegative=True)

    degree = 0
    for indexed in expr.atoms(sp.Indexed):
        if str(indexed.base) != "U" or len(indexed.indices) != 1:
            raise ValueError(f"Unsupported indexed expression: {indexed}")
        lag = sp.simplify(n - indexed.indices[0])
        if not lag.is_Integer or int(lag) < 1:
            raise ValueError(f"Unsupported recurrence lag: {indexed}")
        degree = max(degree, int(lag))
    return degree


def sympy_to_poly(expr, recurrence_depth):
    expr = sp.expand(_sympify_formula(expr))
    variables = sp.symbols(f"x0:{recurrence_depth + 1}")
    n = next(
        (symbol for symbol in expr.free_symbols if symbol.name == "n"),
        sp.Symbol("n", integer=True, nonnegative=True),
    )
    replacements = {n: variables[0]}
    for indexed in expr.atoms(sp.Indexed):
        if str(indexed.base) != "U" or len(indexed.indices) != 1:
            raise ValueError(f"Unsupported indexed expression: {indexed}")
        lag = sp.simplify(n - indexed.indices[0])
        if not lag.is_Integer:
            raise ValueError(f"Non-constant recurrence lag: {indexed}")
        lag = int(lag)
        if not 1 <= lag <= recurrence_depth:
            raise ValueError(f"Lag outside 1..{recurrence_depth}: {indexed}")
        replacements[indexed] = variables[lag]

    polynomial = sp.Poly(sp.expand(expr.xreplace(replacements)), *variables, domain=sp.ZZ)
    return tuple(
        sorted((tuple(powers), int(coefficient)) for powers, coefficient in polynomial.terms())
    )


def parse_formula(formula, recurrence_depth):
    return sympy_to_poly(formula, recurrence_depth)


def poly_to_string(poly):
    """Serialize a sparse polynomial in a compact canonical form."""
    if not poly:
        return "0"

    names = ("n",) + tuple(f"U[n - {lag}]" for lag in range(1, len(poly[0][0])))
    pieces = []
    for powers, coefficient in sorted(
        poly, key=lambda term: (sum(term[0]), term[0]), reverse=True
    ):
        factors = [
            name if power == 1 else f"{name}**{power}"
            for name, power in zip(names, powers)
            if power
        ]
        magnitude = abs(coefficient)
        body = " * ".join(factors)
        if not body:
            body = str(magnitude)
        elif magnitude != 1:
            body = f"{magnitude} * {body}"

        if not pieces:
            pieces.append(f"-{body}" if coefficient < 0 else body)
        else:
            pieces.append(f"{'-' if coefficient < 0 else '+'} {body}")
    return " ".join(pieces)


def convert_to_sympy(tokens, recurrence_depth=3):
    n = sp.Symbol("n", integer=True, nonnegative=True)
    U = sp.IndexedBase("U")
    locals_ = {f"U{lag}": U[n - lag] for lag in range(1, recurrence_depth + 1)}
    locals_["n"] = n
    return sp.sympify("".join(tokens), locals=locals_)


def rollout_prefix(poly, initial_terms, recurrence_depth, n_terms, max_digits=15):
    """Return the valid prefix, retaining terms produced before an explosion."""
    values = list(map(int, initial_terms))
    if len(values) != recurrence_depth:
        raise ValueError("initial_terms must contain recurrence_depth values")
    for rank in range(recurrence_depth, n_terms):
        state = [rank]
        state.extend(values[rank - lag] for lag in range(1, recurrence_depth + 1))
        value = eval_poly(poly, state)
        if max_digits is not None and len(str(value)) > max_digits:
            break
        values.append(value)
    return tuple(values)


def rollout(poly, initial_terms, recurrence_depth, n_terms, max_digits=15):
    values = rollout_prefix(poly, initial_terms, recurrence_depth, n_terms, max_digits)
    return values if len(values) == n_terms else None


@dataclass(frozen=True)
class Candidate:
    poly: tuple
    cost: int
    syntax: str


def _insert_lexicographic(mapping, poly, syntax):
    current = mapping.get(poly)
    if current is None or syntax < current:
        mapping[poly] = syntax


@lru_cache(maxsize=None)
def candidate_bank(recurrence_depth, max_cost=7, constant_min=-9, constant_max=9):
    """Enumerate one AST-shortlex representative per exact polynomial."""
    if max_cost < 1 or max_cost % 2 == 0:
        raise ValueError("max_cost must be a positive odd integer")

    n_vars = recurrence_depth + 1
    names = ("n",) + tuple(f"U[n - {lag}]" for lag in range(1, recurrence_depth + 1))
    exact = {cost: {} for cost in range(1, max_cost + 1, 2)}
    best_cost = {}

    for value in range(constant_min, constant_max + 1):
        _insert_lexicographic(exact[1], constant_poly(value, n_vars), str(value))
    for index, name in enumerate(names):
        _insert_lexicographic(exact[1], variable_poly(index, n_vars), name)
    best_cost.update({poly: 1 for poly in exact[1]})

    for cost in range(3, max_cost + 1, 2):
        current = exact[cost]
        for left_cost in range(1, cost - 1, 2):
            right_cost = cost - 1 - left_cost
            for left_poly, left_syntax in exact[left_cost].items():
                for right_poly, right_syntax in exact[right_cost].items():
                    if left_poly <= right_poly:
                        a, b = sorted((left_syntax, right_syntax))
                        _insert_lexicographic(current, add_poly(left_poly, right_poly), f"({a} + {b})")
                        _insert_lexicographic(current, mul_poly(left_poly, right_poly), f"({a} * {b})")
                    _insert_lexicographic(
                        current,
                        sub_poly(left_poly, right_poly),
                        f"({left_syntax} - {right_syntax})",
                    )
        for poly in tuple(current):
            if poly in best_cost:
                del current[poly]
            else:
                best_cost[poly] = cost

    return tuple(
        sorted(
            (
                Candidate(poly, cost, syntax)
                for cost, expressions in exact.items()
                for poly, syntax in expressions.items()
            ),
            key=lambda candidate: (candidate.cost, candidate.syntax),
        )
    )


@lru_cache(maxsize=None)
def candidate_index(recurrence_depth, max_cost=7, constant_min=-9, constant_max=9):
    return {
        candidate.poly: index
        for index, candidate in enumerate(
            candidate_bank(recurrence_depth, max_cost, constant_min, constant_max)
        )
    }


def poly_degree(poly):
    return max(
        (index for powers, _ in poly for index in range(1, len(powers)) if powers[index]),
        default=0,
    )


@lru_cache(maxsize=None)
def candidate_pool(recurrence_depth, max_cost=7):
    return tuple(
        candidate
        for candidate in candidate_bank(recurrence_depth, max_cost)
        if poly_degree(candidate.poly) == recurrence_depth
    )


def bank_fingerprint(recurrence_depth, max_cost, constant_min=-9, constant_max=9):
    digest = hashlib.sha256()
    for candidate in candidate_bank(recurrence_depth, max_cost, constant_min, constant_max):
        digest.update(repr((candidate.poly, candidate.cost, candidate.syntax)).encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Identification:
    candidate: Candidate
    terms: tuple
    n_visible: int


def _status_for_initial(
    initial_terms,
    recurrence_depth,
    min_visible,
    max_visible,
    max_cost,
    max_digits,
):
    bank = candidate_bank(recurrence_depth, max_cost)
    trajectories = [
        rollout_prefix(candidate.poly, initial_terms, recurrence_depth, max_visible, max_digits)
        for candidate in bank
    ]
    status = np.zeros(len(bank), dtype=np.uint8)
    for n_visible in range(min_visible, max_visible + 1):
        groups = {}
        for index, (candidate, terms) in enumerate(zip(bank, trajectories)):
            if len(terms) < n_visible:
                continue
            signature = terms[recurrence_depth:n_visible]
            group = groups.get(signature)
            if group is None or candidate.cost < group[0]:
                groups[signature] = [candidate.cost, index, 1]
            elif candidate.cost == group[0]:
                group[2] += 1
        for _, index, count in groups.values():
            if count == 1 and not status[index]:
                status[index] = n_visible
    return status


def identify_online(
    target_poly,
    initial_terms,
    recurrence_depth,
    min_visible=8,
    max_visible=16,
    max_cost=7,
    max_digits=15,
):
    bank = candidate_bank(recurrence_depth, max_cost)
    index = candidate_index(recurrence_depth, max_cost).get(target_poly)
    if index is None:
        return None, "outside_bank"
    target_terms = rollout_prefix(
        target_poly, initial_terms, recurrence_depth, max_visible, max_digits
    )
    if len(target_terms) < min_visible:
        return None, "explosion"
    status = _status_for_initial(
        tuple(initial_terms),
        recurrence_depth,
        min_visible,
        min(max_visible, len(target_terms)),
        max_cost,
        max_digits,
    )
    n_visible = int(status[index])
    if not n_visible:
        return None, "ambiguous"
    return Identification(bank[index], target_terms[:n_visible], n_visible), "accepted"


def _uniqueness_row_paths(
    cache_dir,
    initial_terms,
    recurrence_depth,
    max_cost,
    min_visible,
    max_visible,
    max_digits,
):
    fingerprint = bank_fingerprint(recurrence_depth, max_cost)
    initial_digest = hashlib.sha256(repr(tuple(initial_terms)).encode()).hexdigest()[:16]
    stem = (
        f"d{recurrence_depth}_c{max_cost}_v{min_visible}-{max_visible}_"
        f"m{max_digits}_{fingerprint}_i{initial_digest}"
    )
    root = Path(cache_dir).expanduser()
    return root / f"{stem}.npy", root / f"{stem}.lock"


def build_uniqueness_row(
    cache_dir,
    initial_terms,
    recurrence_depth,
    min_visible=8,
    max_visible=16,
    max_cost=7,
    max_digits=15,
    force=False,
):
    """Build one cached row; storage is linear rather than 19**depth."""
    data_path, lock_path = _uniqueness_row_paths(
        cache_dir,
        tuple(initial_terms),
        recurrence_depth,
        max_cost,
        min_visible,
        max_visible,
        max_digits,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists() and not force:
        return np.load(data_path, mmap_mode="r")

    import fcntl

    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if data_path.exists() and not force:
            return np.load(data_path, mmap_mode="r")
        status = _status_for_initial(
            tuple(initial_terms),
            recurrence_depth,
            min_visible,
            max_visible,
            max_cost,
            max_digits,
        )
        temp_path = data_path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(temp_path, status)
        os.replace(temp_path, data_path)
    return np.load(data_path, mmap_mode="r")


def identify_cached(
    target_poly,
    initial_terms,
    recurrence_depth,
    cache_dir,
    min_visible=8,
    max_visible=16,
    max_cost=7,
    max_digits=15,
):
    bank = candidate_bank(recurrence_depth, max_cost)
    index = candidate_index(recurrence_depth, max_cost).get(target_poly)
    if index is None:
        return None, "outside_bank"
    status = build_uniqueness_row(
        cache_dir,
        tuple(initial_terms),
        recurrence_depth,
        min_visible,
        max_visible,
        max_cost,
        max_digits,
    )
    n_visible = int(status[index])
    if not n_visible:
        return None, "ambiguous_or_invalid"
    terms = rollout(target_poly, initial_terms, recurrence_depth, n_visible, max_digits)
    if terms is None:
        return None, "explosion"
    return Identification(bank[index], terms, n_visible), "accepted"


class Sequence:
    def __init__(self, formula, initial_elem=None, initial_min=-9, initial_max=9):
        self.rec_formula = _sympify_formula(formula)
        self.degree = formula_degree(self.rec_formula)
        self.poly = sympy_to_poly(self.rec_formula, self.degree)
        if initial_elem is None:
            self.first_elem = list(
                map(int, np.random.randint(initial_min, initial_max + 1, size=self.degree))
            )
        elif len(initial_elem) >= self.degree:
            self.first_elem = list(map(int, initial_elem[: self.degree]))
        else:
            raise ValueError(f"Degree {self.degree} requires at least {self.degree} initial terms")

    def __repr__(self):
        return (
            f"Sequence(formula={self.rec_formula}, degree={self.degree}, "
            f"initial_terms={self.first_elem})"
        )

    def U_n(self, predecessors, rank):
        state = [rank]
        state.extend(predecessors[rank - lag] for lag in range(1, self.degree + 1))
        return eval_poly(self.poly, state)

    def n_first_elem(self, n, max_terms_len=15):
        if n < self.degree:
            raise ValueError("n must be at least the recurrence degree")
        values = rollout(self.poly, self.first_elem, self.degree, n, max_terms_len)
        return list(values) if values is not None else []


@dataclass
class SequenceConfig(Config):
    mode: str = "simple"
    recurrence_depth: int = 1
    n_visible_terms: int = 8
    max_visible_terms: int = 16
    max_terms_len: int = 15
    min_depth_grammar: int = 2
    max_depth_grammar: int = 3
    canonical_max_cost: int = 7
    initial_min: int = -9
    initial_max: int = 9
    sampling: str = "bank"
    uniqueness_cache_dir: str = "~/.cache/sequential_induction"
    use_uniqueness_cache: bool = True
    max_generation_attempts: int = 10_000

    def apply_difficulty(self, level):
        self.recurrence_depth = sround(self.recurrence_depth + level)
        self.n_visible_terms = sround(self.n_visible_terms + 2 * level)
        self.max_visible_terms = max(self.max_visible_terms, self.n_visible_terms + 8)
        self.min_depth_grammar = sround(self.min_depth_grammar + 0.5 * level)
        self.max_depth_grammar = sround(self.max_depth_grammar + level)


class SequentialInduction(Task):
    summary = "Infer the canonical recurrence in a bounded polynomial DSL."
    config_cls = SequenceConfig

    def __init__(self, config=None):
        super().__init__(config=config or SequenceConfig())
        if self.config.mode != "simple":
            raise NotImplementedError("Only canonical polynomial simple mode is supported")
        if self.config.sampling not in {"bank", "grammar"}:
            raise ValueError("sampling must be 'bank' or 'grammar'")

    def one_shot_sympy_generate(self):
        rule = Sequence_cfg(self.config.mode, self.config.recurrence_depth)
        tokens = generate(
            rule,
            depth=self.config.max_depth_grammar,
            min_depth=self.config.min_depth_grammar,
        ) @ "eq"
        return convert_to_sympy(tokens, self.config.recurrence_depth)

    def _identify(self, target_poly, initial_terms, degree):
        kwargs = dict(
            target_poly=target_poly,
            initial_terms=tuple(initial_terms),
            recurrence_depth=degree,
            min_visible=self.config.n_visible_terms,
            max_visible=self.config.max_visible_terms,
            max_cost=self.config.canonical_max_cost,
            max_digits=self.config.max_terms_len,
        )
        if self.config.use_uniqueness_cache:
            return identify_cached(cache_dir=self.config.uniqueness_cache_dir, **kwargs)
        return identify_online(**kwargs)

    def generate_entry(self):
        reasons = Counter()
        for _ in range(self.config.max_generation_attempts):
            try:
                if self.config.sampling == "bank":
                    degree = int(np.random.randint(self.config.recurrence_depth + 1))
                    pool = candidate_pool(degree, self.config.canonical_max_cost)
                    target_poly = pool[int(np.random.randint(len(pool)))].poly
                else:
                    formula = self.one_shot_sympy_generate()
                    degree = formula_degree(formula)
                    target_poly = sympy_to_poly(formula, degree)
                initial_terms = list(
                    map(
                        int,
                        np.random.randint(
                            self.config.initial_min,
                            self.config.initial_max + 1,
                            size=degree,
                        ),
                    )
                )
                identification, reason = self._identify(target_poly, initial_terms, degree)
            except (ValueError, TypeError, sp.SympifyError, sp.PolynomialError):
                reasons["invalid_formula"] += 1
                continue
            if identification is None:
                reasons[reason] += 1
                continue
            generated = identification.terms[degree:]
            if len(set(generated)) <= 1:
                reasons["constant"] += 1
                continue
            metadata = {
                "first elements": list(identification.terms),
                "degree of recursion": degree,
                "initial terms": initial_terms,
                "canonical cost": identification.candidate.cost,
                "canonical max cost": self.config.canonical_max_cost,
            }
            return Entry(metadata=metadata, answer=poly_to_string(target_poly))
        raise RuntimeError(
            "Could not generate a canonical unique sequence after "
            f"{self.config.max_generation_attempts} attempts: {dict(reasons)}"
        )

    def verify(self, y_pred, y_truth, initial_element=None):
        del initial_element
        try:
            degree = formula_degree(y_truth)
            predicted_poly = parse_formula(y_pred, degree)
            truth_poly = parse_formula(y_truth, degree)
            return predicted_poly == truth_poly and predicted_poly in candidate_index(
                degree, self.config.canonical_max_cost
            )
        except Exception:
            return False

    def score_answer(self, answer, entry):
        degree = entry.metadata["degree of recursion"]
        max_cost = entry.metadata.get(
            "canonical max cost",
            entry.metadata.get("_config", {}).get(
                "canonical_max_cost", SequenceConfig.canonical_max_cost
            ),
        )
        try:
            text = str(answer).strip()
            assignment = re.fullmatch(r"U\s*\[\s*n\s*\]\s*=\s*(.+)", text, re.DOTALL)
            predicted_poly = parse_formula(assignment.group(1) if assignment else text, degree)
            expected_poly = parse_formula(entry.answer, degree)
            return float(
                predicted_poly == expected_poly
                and predicted_poly in candidate_index(degree, max_cost)
            )
        except Exception:
            return 0.0

    def render_prompt(self, metadata):
        degree = metadata["degree of recursion"]
        refs = (
            "n"
            if degree == 0
            else "U[n - 1] and n"
            if degree == 1
            else f"U[n - 1] ... U[n - {degree}] and n"
        )
        lines = [
            f"Infer U[n]. Max recurrence degree: {degree}. Ops: +, -, *.",
            f"Use {refs}. Give the simplified polynomial RHS.",
            f"Sequence: {metadata['first elements']}",
        ]
        if degree:
            lines.append(f"Initial terms: {metadata['initial terms']}")
        lines.append("The answer is the RHS only.")
        return "\n".join(lines)

    def deduplication_key(self, problem):
        return problem.answer, tuple(problem.metadata["first elements"])


def precompute_level(config=None, initial_terms_by_degree=None):
    """Warm candidate banks and, optionally, specified uniqueness rows."""
    config = config or SequenceConfig()
    banks = {
        degree: candidate_bank(degree, config.canonical_max_cost)
        for degree in range(config.recurrence_depth + 1)
    }
    for degree, rows in (initial_terms_by_degree or {}).items():
        for initial_terms in rows:
            build_uniqueness_row(
                config.uniqueness_cache_dir,
                initial_terms,
                degree,
                config.n_visible_terms,
                config.max_visible_terms,
                config.canonical_max_cost,
                config.max_terms_len,
            )
    return banks
