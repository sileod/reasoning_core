import itertools
import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


@dataclass
class BooleanPropagationSearchConfig(Config):
    n_vars: int = 5
    n_clauses: int = 8
    max_attempts: int = 500

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + 0.8 * level)
        self.n_clauses = sround(self.n_clauses + 1.8 * level)
        self.max_attempts = sround(self.max_attempts + 80 * level)


def _state(clause, values):
    open_ = []
    for lit in clause:
        var = abs(lit)
        if var not in values:
            open_.append(lit)
        elif values[var] == (lit > 0):
            return True, []
    return False, open_


def _first_model(clauses, n_vars):
    stats = {"unit": 0, "conflicts": 0, "backtracks": 0}

    def visit(values):
        values = dict(values)
        while True:
            units = []
            for clause in clauses:
                sat, open_ = _state(clause, values)
                if sat:
                    continue
                if not open_:
                    stats["conflicts"] += 1
                    return None
                if len(open_) == 1:
                    lit = open_[0]
                    units.append((abs(lit), lit > 0))
            if not units:
                break
            var, value = min(units)
            if var in values and values[var] != value:
                stats["conflicts"] += 1
                return None
            if var not in values:
                values[var] = value
                stats["unit"] += 1
        if len(values) == n_vars:
            return values
        var = min(set(range(1, n_vars + 1)) - values.keys())
        for value in (False, True):
            found = visit({**values, var: value})
            if found is not None:
                return found
            stats["backtracks"] += 1
        return None

    return visit({}), stats


def _satisfies(clauses, values):
    return all(any(values[abs(lit) - 1] == (lit > 0) for lit in clause) for clause in clauses)


def _multiple_models(clauses, n_vars):
    return sum(_satisfies(clauses, xs) for xs in itertools.product((False, True), repeat=n_vars)) >= 2


def _clause_text(clause):
    return "(" + " or ".join(("" if lit > 0 else "not ") + f"x{abs(lit)}" for lit in clause) + ")"


class BooleanPropagationSearch(GeneratedMixin, Task):
    summary = "Find the canonical first Boolean model under propagation and backtracking."
    config_cls = BooleanPropagationSearchConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            clauses = []
            for _ in range(cfg.n_clauses):
                variables = random.sample(range(1, cfg.n_vars + 1), random.choice((2, 3)))
                clause = tuple(v if random.random() < 0.5 else -v for v in variables)
                if clause not in clauses:
                    clauses.append(clause)
            model, stats = _first_model(clauses, cfg.n_vars)
            if model is None or stats["conflicts"] < 1 or stats["unit"] < 1 or not _multiple_models(clauses, cfg.n_vars):
                continue
            answer = " ".join("T" if model[i] else "F" for i in range(1, cfg.n_vars + 1))
            return Entry(metadata=edict(clauses=[list(c) for c in clauses], n_vars=cfg.n_vars, stats=stats), answer=answer)
        raise RuntimeError("Failed to generate a nontrivial Boolean propagation instance")

    def render_prompt(self, metadata):
        formula = " and ".join(_clause_text(c) for c in metadata.clauses)
        return (
            f"Formula: {formula}\n"
            "Choose unassigned variables x1,x2,... in order and try False before True. Before each choice, repeatedly assign any value forced by a one-unassigned-literal clause; if several are forced, use the smallest variable first. Backtrack on contradiction.\n"
            f"What is the first satisfying assignment found? The answer is {metadata.n_vars} space-separated T/F values for x1..x{metadata.n_vars}."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).upper().replace(",", " ").split())
        return float(norm(answer) == norm(entry.answer))

    def balancing_key(self, problem):
        return min(4, problem.metadata.stats["conflicts"])
