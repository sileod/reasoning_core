import random
import re
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


ATTRIBUTES = {
    "shape": ("circle", "triangle", "square", "diamond"),
    "count": ("1", "2", "3", "4"),
    "orientation": ("north", "east", "south", "west"),
    "fill": ("open", "filled"),
}


def _rules(k):
    rules = [
        ("left", lambda a, b, k=k: a),
        ("right", lambda a, b, k=k: b),
        ("min", lambda a, b, k=k: min(a, b)),
        ("max", lambda a, b, k=k: max(a, b)),
    ]
    rules += [(f"add+{c}", lambda a, b, c=c, k=k: (a + b + c) % k) for c in range(k)]
    if k & (k - 1) == 0:
        rules += [(f"xor+{c}", lambda a, b, c=c, k=k: a ^ b ^ c) for c in range(k)]
    return rules


def _satisfies(grid, rule):
    return all(rule(row[0], row[1]) == row[2] for row in grid) and all(
        rule(grid[0][j], grid[1][j]) == grid[2][j] for j in range(3)
    )


def _complete_values(partial, missing, k):
    values = set()
    survivors = {}
    for value in range(k):
        grid = [row[:] for row in partial]
        grid[missing[0]][missing[1]] = value
        names = [name for name, rule in _rules(k) if _satisfies(grid, rule)]
        if names:
            values.add(value)
            survivors[value] = names
    return values, survivors


def _make_grid(rule, k):
    grid = [[None] * 3 for _ in range(3)]
    for i in range(2):
        for j in range(2):
            grid[i][j] = random.randrange(k)
    for i in range(2):
        grid[i][2] = rule(grid[i][0], grid[i][1])
    for j in range(2):
        grid[2][j] = rule(grid[0][j], grid[1][j])
    grid[2][2] = rule(grid[2][0], grid[2][1])
    return grid if rule(grid[0][2], grid[1][2]) == grid[2][2] else None


def _cell_text(values, attrs):
    return " ".join(f"{name}={ATTRIBUTES[name][values[name]]}" for name in attrs)


@dataclass
class MatrixInductionConfig(Config):
    n_attributes: int = 1
    hard_position_p: float = 0.15
    hard_rule_p: float = 0.35

    def apply_difficulty(self, level):
        self.n_attributes = sround(self.n_attributes + 0.6 * level)
        self.hard_position_p = min(0.9, self.hard_position_p + 0.12 * level)
        self.hard_rule_p = min(0.9, self.hard_rule_p + 0.1 * level)


class MatrixInduction(GeneratedMixin, Task):
    summary = "Infer a missing multi-attribute matrix cell under a certified finite rule family."
    config_cls = MatrixInductionConfig

    def generate_entry(self):
        n_attributes = min(4, max(1, int(self.config.n_attributes)))
        attrs = random.sample(list(ATTRIBUTES), n_attributes)

        for _ in range(300):
            missing = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)]) if random.random() < self.config.hard_position_p else (2, 2)
            grids = {}
            hidden_rules = {}
            survivor_counts = {}
            ok = True

            for attr in attrs:
                k = len(ATTRIBUTES[attr])
                bank = _rules(k)
                hard = [x for x in bank if x[0].startswith(("add", "xor"))]
                name, rule = random.choice(hard if hard and random.random() < self.config.hard_rule_p else bank)
                grid = _make_grid(rule, k)
                if grid is None or len({x for row in grid for x in row}) < 2:
                    ok = False
                    break
                partial = [row[:] for row in grid]
                partial[missing[0]][missing[1]] = None
                values, survivors = _complete_values(partial, missing, k)
                if values != {grid[missing[0]][missing[1]]}:
                    ok = False
                    break
                grids[attr] = grid
                hidden_rules[attr] = name
                survivor_counts[attr] = len(survivors[grid[missing[0]][missing[1]]])

            if not ok:
                continue

            cells = []
            for i in range(3):
                row = []
                for j in range(3):
                    if (i, j) == missing:
                        row.append("?")
                    else:
                        row.append(_cell_text({a: grids[a][i][j] for a in attrs}, attrs))
                cells.append(row)

            gold_values = {a: grids[a][missing[0]][missing[1]] for a in attrs}
            answer = _cell_text(gold_values, attrs)
            return Entry(
                metadata=edict(
                    attributes=attrs,
                    cells=cells,
                    missing=missing,
                    hidden_rules=hidden_rules,
                    survivor_counts=survivor_counts,
                ),
                answer=answer,
            )
        raise RuntimeError("MatrixInduction: could not build a uniquely completable instance")

    def render_prompt(self, m):
        domains = "\n".join(f"- {a}: " + ", ".join(ATTRIBUTES[a]) for a in m.attributes)
        rows = "\n".join(" | ".join(row) for row in m.cells)
        return (
            "Complete the missing cell of the 3x3 matrix. Each attribute is independent and uses one fixed rule "
            "for every row and every column. Encode each listed domain by indices 0,1,... in the shown order. "
            "For a row or column with encoded values a,b,c, c is obtained from a,b by one of: left=a; right=b; "
            "min=min(a,b); max=max(a,b); add+t=(a+b+t) mod k for some t; xor+t=a xor b xor t for some t "
            "(xor is used only for power-of-two domain sizes). Different attributes may use different rules.\n"
            f"Domains:\n{domains}\nMatrix:\n{rows}\n"
            "The answer is the missing cell written with exactly the displayed attribute names as name=value pairs."
        )

    def score_answer(self, answer, entry):
        def parse(text):
            out = {}
            for name in entry.metadata["attributes"]:
                m = re.search(rf"\b{name}\s*=\s*([A-Za-z0-9_-]+)", str(text), re.I)
                if not m:
                    return None
                out[name] = m.group(1).lower()
            return out
        return float(parse(answer) == parse(entry.answer))

    def balancing_key(self, problem):
        return tuple(problem.metadata.hidden_rules.values())
