import math
import random
from dataclasses import dataclass

import sympy as sp

from reasoning_core.template import Entry, Config, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


@dataclass
class VariableEliminationConfig(Config):
    n_vars: int = 4
    coefficient_magnitude: int = 5
    eliminate: int = 2
    max_attempts: int = 300

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + 0.45 * level)
        self.coefficient_magnitude = sround(self.coefficient_magnitude + 0.6 * level)
        self.eliminate = sround(self.eliminate + 0.35 * level)
        self.max_attempts = sround(self.max_attempts + 30 * level)


def _primitive_row(row):
    values = [int(x) for x in row]
    g = 0
    for x in values:
        g = math.gcd(g, abs(x))
    if g:
        values = [x // g for x in values]
    first = next((x for x in values if x), 1)
    if first < 0:
        values = [-x for x in values]
    return values


def _eliminate_rows(rows, n_eliminate):
    rows = [list(map(int, r)) for r in rows]
    history = []
    for col in range(n_eliminate):
        pivot_i = next((i for i in range(col, len(rows)) if rows[i][col]), None)
        if pivot_i is None:
            return None, history
        rows[col], rows[pivot_i] = rows[pivot_i], rows[col]
        pivot = rows[col]
        for i in range(col + 1, len(rows)):
            if rows[i][col] == 0:
                continue
            a, b = pivot[col], rows[i][col]
            rows[i] = _primitive_row([a * x - b * y for x, y in zip(rows[i], pivot)])
        history.append([r[:] for r in rows])
    return rows, history


def _equation_text(row, names):
    lhs = []
    for c, name in zip(row[:-1], names):
        if not c:
            continue
        sign = "+" if c > 0 else "-"
        term = name if abs(c) == 1 else f"{abs(c)}*{name}"
        if not lhs:
            lhs.append(term if c > 0 else "-" + term)
        else:
            lhs.append(f" {sign} {term}")
    return ("".join(lhs) or "0") + f" = {row[-1]}"


class VariableEliminationMixin(GeneratedMixin):
    summary = "Execute deterministic fraction-free elimination and report a compact residual equation."
    config_cls = VariableEliminationConfig

    def generate_entry(self):
        cfg = self.config
        n = max(3, cfg.n_vars)
        k = min(max(1, cfg.eliminate), n - 2)
        mag = cfg.coefficient_magnitude
        for _ in range(cfg.max_attempts):
            matrix = sp.Matrix([[random.randint(-mag, mag) for _ in range(n)] for _ in range(n)])
            if matrix.det() == 0:
                continue
            rhs = [random.randint(-2 * mag, 2 * mag) for _ in range(n)]
            rows = [[int(matrix[i, j]) for j in range(n)] + [rhs[i]] for i in range(n)]
            out, history = _eliminate_rows(rows, k)
            if out is None:
                continue
            target = random.randrange(k, n)
            row = _primitive_row(out[target])
            if sum(x != 0 for x in row[k:-1]) < 2:
                continue
            names = [f"x{i + 1}" for i in range(n)]
            answer = _equation_text(row, names)
            metadata = edict(rows=rows, names=names, eliminate=k, target=target + 1,
                             answer_row=row, n_vars=n)
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("Failed to generate elimination instance")

    def render_prompt(self, metadata):
        equations = "\n".join(_equation_text(r, metadata.names) for r in metadata.rows)
        eliminated = ", ".join(metadata.names[:metadata.eliminate])
        return (
            f"Equations:\n{equations}\n"
            f"Eliminate {eliminated} in that order. For each variable, use the first remaining equation with a nonzero coefficient as pivot. "
            "For every later row with coefficient b and pivot coefficient a, replace it by a*row - b*pivot; then divide the entire row by the gcd of its integer coefficients and make its first nonzero coefficient positive.\n"
            f"After these eliminations, what is row {metadata.target}? The answer is one simplified equation."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).replace(" ", "").split())
        return float(norm(answer) == norm(entry.answer))

    def balancing_key(self, problem):
        return problem.metadata.eliminate
