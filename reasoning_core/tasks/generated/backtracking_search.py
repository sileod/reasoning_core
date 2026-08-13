import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


@dataclass
class BacktrackingSearchConfig(Config):
    n_vars: int = 5
    domain_size: int = 4
    n_constraints: int = 7
    min_backtracks: int = 1
    max_attempts: int = 400

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + 0.7 * level)
        self.domain_size = sround(self.domain_size + 0.25 * level)
        self.n_constraints = sround(self.n_constraints + 1.4 * level)
        self.min_backtracks = sround(self.min_backtracks + 0.4 * level)
        self.max_attempts = sround(self.max_attempts + 60 * level)


def _constraint_ok(constraint, a, b):
    kind, c = constraint
    if kind == "neq":
        return a != b
    if kind == "lt":
        return a < b
    if kind == "gt":
        return a > b
    if kind == "sum_ne":
        return a + b != c
    if kind == "diff_ne":
        return a - b != c
    raise ValueError(kind)


def _forward_search(n_vars, domain_size, constraints):
    by_pair = {}
    for i, j, kind, c in constraints:
        by_pair.setdefault((i, j), []).append((kind, c))
    stats = {"nodes": 0, "backtracks": 0, "prunes": 0}

    def pair_ok(i, a, j, b):
        tests = by_pair.get((i, j), [])
        rev = by_pair.get((j, i), [])
        return all(_constraint_ok(t, a, b) for t in tests) and all(_constraint_ok(t, b, a) for t in rev)

    def rec(assignment, domains):
        if len(assignment) == n_vars:
            return dict(assignment)
        var = len(assignment)
        for value in sorted(domains[var]):
            stats["nodes"] += 1
            if any(not pair_ok(i, assignment[i], var, value) for i in assignment):
                stats["backtracks"] += 1
                continue
            child_domains = [set(d) for d in domains]
            child_domains[var] = {value}
            failed = False
            for j in range(var + 1, n_vars):
                allowed = {b for b in child_domains[j] if pair_ok(var, value, j, b)}
                stats["prunes"] += len(child_domains[j]) - len(allowed)
                child_domains[j] = allowed
                if not allowed:
                    failed = True
                    break
            if failed:
                stats["backtracks"] += 1
                continue
            child = dict(assignment)
            child[var] = value
            out = rec(child, child_domains)
            if out is not None:
                return out
            stats["backtracks"] += 1
        return None

    domains = [set(range(1, domain_size + 1)) for _ in range(n_vars)]
    return rec({}, domains), stats


def _count_csp_models(n_vars, domain_size, constraints, limit=2):
    by_pair = {}
    for i, j, kind, c in constraints:
        by_pair.setdefault((i, j), []).append((kind, c))

    def pair_ok(i, a, j, b):
        return (
            all(_constraint_ok(t, a, b) for t in by_pair.get((i, j), []))
            and all(_constraint_ok(t, b, a) for t in by_pair.get((j, i), []))
        )

    count = 0

    def rec(values):
        nonlocal count
        if count >= limit:
            return
        var = len(values)
        if var == n_vars:
            count += 1
            return
        for value in range(1, domain_size + 1):
            if all(pair_ok(i, values[i], var, value) for i in range(var)):
                rec(values + [value])
                if count >= limit:
                    return

    rec([])
    return count


def _render_constraint(c):
    i, j, kind, k = c
    x, y = f"X{i + 1}", f"X{j + 1}"
    return {
        "neq": f"{x} != {y}",
        "lt": f"{x} < {y}",
        "gt": f"{x} > {y}",
        "sum_ne": f"{x} + {y} != {k}",
        "diff_ne": f"{x} - {y} != {k}",
    }[kind]


class BacktrackingSearchMixin(GeneratedMixin):
    summary = "Find the first finite-domain solution under deterministic backtracking with forward checking."
    config_cls = BacktrackingSearchConfig

    def generate_entry(self):
        cfg = self.config
        kinds = ("neq", "lt", "gt", "sum_ne", "diff_ne")
        for _ in range(cfg.max_attempts):
            constraints = []
            for _ in range(cfg.n_constraints):
                i, j = sorted(random.sample(range(cfg.n_vars), 2))
                kind = random.choice(kinds)
                c = random.randint(2, 2 * cfg.domain_size) if kind == "sum_ne" else random.randint(-cfg.domain_size + 1, cfg.domain_size - 1)
                item = (i, j, kind, c)
                if item not in constraints:
                    constraints.append(item)
            model, stats = _forward_search(cfg.n_vars, cfg.domain_size, constraints)
            if model is None or stats["backtracks"] < cfg.min_backtracks or stats["prunes"] < 2:
                continue
            if _count_csp_models(cfg.n_vars, cfg.domain_size, constraints) < 2:
                continue
            answer = " ".join(str(model[i]) for i in range(cfg.n_vars))
            metadata = edict(constraints=constraints, n_vars=cfg.n_vars, domain_size=cfg.domain_size, stats=stats)
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("Failed to generate a nontrivial backtracking instance")

    def render_prompt(self, metadata):
        constraints = "; ".join(_render_constraint(c) for c in metadata.constraints)
        return (
            f"Variables X1..X{metadata.n_vars} each range over 1..{metadata.domain_size}.\n"
            f"Constraints: {constraints}\n"
            "Search variables in order X1,X2,... and values in increasing order. After each assignment, "
            "remove from every later domain values that violate a constraint with the new assignment; "
            "backtrack immediately if a domain becomes empty.\n"
            "What is the first complete solution found? The answer is the space-separated values of X1..Xn."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).replace(",", " ").split())
        return float(norm(answer) == norm(entry.answer))

    def balancing_key(self, problem):
        return min(4, problem.metadata.stats["backtracks"])
