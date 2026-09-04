import random
import re
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'spreadsheet_formula_dependency (draw 1 of 1)',
 'hypothesis': 'HV-035',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/spreadsheet_formula_dependency',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 780191338,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class SpreadsheetFormulaConfig(Config):
    n_cells: int = 10
    max_val: int = 15
    max_edits: int = 2
    max_deps: int = 4

    def apply_difficulty(self, level):
        self.n_cells = sround(self.n_cells + 3 * level)
        self.max_val = sround(self.max_val + 4 * level)
        self.max_edits = sround(self.max_edits + (level >= 2))
        self.max_deps = sround(self.max_deps + (level >= 3))


def _cell_names(n_cells):
    return ["A%d" % (i + 1) for i in range(n_cells)]


REF_RE = re.compile(r"\b[A-Z]\d+\b")
RANGE_RE = re.compile(r"([A-Z])(\d+):([A-Z])(\d+)")


def _refs_of(formula):
    refs = set()
    s = str(formula)
    refs.update(REF_RE.findall(s))
    refs.discard("")
    for m in RANGE_RE.finditer(s):
        c1, r1 = m.group(1), int(m.group(2))
        c2, r2 = m.group(3), int(m.group(4))
        if c1 == c2:
            for row in range(r1, r2 + 1):
                refs.add("%s%d" % (c1, row))
    return refs


def _expand_ranges(formula, cell_values):
    def repl(m):
        c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if c1 != c2:
            return m.group(0)
        vals = []
        for row in range(r1, r2 + 1):
            nm = "%s%d" % (c1, row)
            if nm in cell_values:
                vals.append(str(cell_values[nm]))
        return ", ".join(vals)
    return RANGE_RE.sub(repl, str(formula))


def _eval_arith(expr):
    expr = expr.strip()
    expr = re.sub(r"\s+", "", expr)
    if not expr:
        raise ValueError("empty expr")

    # tokenize numbers separated by + and *
    parts = re.split(r"([+*])", expr)
    # evaluate * first
    stack = []
    i = 0
    while i < len(parts):
        if parts[i] == "":
            i += 1
            continue
        if parts[i] in ("+",):
            stack.append(parts[i])
            i += 1
        elif parts[i] == "*":
            left = stack.pop()
            right = parts[i + 1]
            if not isinstance(left, int):
                left = int(left)
            if not isinstance(right, int):
                right = int(right)
            stack.append(left * right)
            i += 2
        else:
            stack.append(int(parts[i]))
            i += 1
    # now sum stack
    total = 0
    for item in stack:
        if item == "+":
            continue
        total += item
    return total


def _solve_formula(formula, cell_values):
    """cell_values: dict name->int for leaf/number cells reachable. formula uses names/numbers/ranges."""
    if isinstance(formula, int):
        return formula
    expr = _expand_ranges(formula, cell_values)
    # substitute all references (single) with their values
    for _ in range(len(cell_values) + 2):
        def repl(m):
            name = m.group(0)
            if name in cell_values:
                return str(cell_values[name])
            return m.group(0)
        new_expr = REF_RE.sub(repl, expr)
        if new_expr == expr:
            break
        expr = new_expr
    # handle min(...), max(...), sum(...)
    expr = expr.strip()
    m = re.fullmatch(r"(max|min|sum)\((.*)\)", expr)
    if m:
        fn = m.group(1)
        inner = m.group(2)
        nums = [int(x) for x in re.findall(r"-?\d+", inner)]
        if fn == "max":
            return max(nums)
        if fn == "min":
            return min(nums)
        return sum(nums)
    return _eval_arith(expr)


class SpreadsheetFormulaDependency(Task):
    summary = "Evaluate spreadsheet formulas over a dependency graph after stated cell edits, including ranges and references, returning requested cell values."

    config_cls = SpreadsheetFormulaConfig

    def _gen(self):
        cfg = self.config
        names = _cell_names(cfg.n_cells)
        n = len(names)

        state = {}
        depends = {}
        for i, name in enumerate(names):
            r = random.random()
            if r < 0.35 or i == 0:
                v = random.randint(1, cfg.max_val)
                state[name] = v
                depends[name] = set()
            else:
                if random.random() < 0.3 and i >= 3:
                    start = random.randint(0, i - 3)
                    length = random.randint(2, min(i - start, cfg.max_deps))
                    block = names[start:start + length]
                    depends[name] = set(block)
                    op = random.choice(["sum", "max", "min"])
                    expr = "%s(%s:%s)" % (op, block[0], block[-1])
                    state[name] = expr
                else:
                    k = random.randint(1, min(cfg.max_deps, i))
                    preds = random.sample(names[:i], k)
                    depends[name] = set(preds)
                    op = random.choice(["+", "max", "min"])
                    dep_exprs = list(preds)
                    if random.random() < 0.3:
                        dep_exprs.append(str(random.randint(1, cfg.max_val)))
                    if op == "+":
                        pre = " + ".join(dep_exprs)
                    else:
                        pre = "%s(%s)" % (op, ", ".join(dep_exprs))
                    state[name] = pre

        formula_cells = [nm for nm in names if not isinstance(state[nm], int)]
        if not formula_cells:
            formula_cells = names
        q = random.choice(formula_cells)

        # apply edits: change some literal cells to new values
        edit_rules = []
        literal_cells = [nm for nm in names if isinstance(state[nm], int)]
        n_edits = min(cfg.max_edits, len(literal_cells))
        picked = random.sample(literal_cells, n_edits)
        for ec in picked:
            newv = random.randint(1, cfg.max_val)
            state[ec] = newv
            edit_rules.append((ec, newv))

        # compute values bottom-up (state[dep] cells are lower index so refs already evaluated)
        cell_values = {}
        for name in names:
            cell_values[name] = _solve_formula(state[name], cell_values)

        answer = cell_values[q]
        assert isinstance(answer, int)
        assert answer >= 0

        # independent verifier: full recompute using memo
        def compute(target):
            memo = {}
            def rec(c):
                if c in memo:
                    return memo[c]
                f = state[c]
                if isinstance(f, int):
                    memo[c] = f
                    return f
                refs = _refs_of(f)
                vals = {}
                for r in refs:
                    if r in state:
                        vals[r] = rec(r)
                memo[c] = _solve_formula(f, vals)
                return memo[c]
            return rec(target)

        for r in _refs_of(q) | {q}:
            if r in state:
                assert compute(r) == cell_values[r]

        lines = ["%s: %s" % (name, state[name]) for name in names]
        edits_desc = ", ".join("%s set to %d" % (c, v) for c, v in edit_rules)

        payload = {"cells": "\n".join(lines)}
        metadata = edict({
            "payload": payload,
            "edits": edits_desc,
            "query": q,
        })
        return Entry(metadata=metadata, answer=str(answer))

    def generate_entry(self):
        while True:
            try:
                return self._gen()
            except (ValueError, RecursionError):
                continue

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        edits = ""
        if metadata.edits:
            edits = ("Afterward the following cells are changed: %s.\n" % metadata.edits)
        return ("Below is a spreadsheet. Each line gives a cell name and either a literal number or "
                "a formula. A formula's value is computed from the current values of the cells it "
                "references; references always point at cells listed above them.\n\n"
                "%s\n"
                "%s"
                "Queries: the value of cell %s.\n\nThe answer is a single integer."
                % (payload, edits, metadata.query))

    def score_answer(self, answer, entry):
        try:
            return float(answer) == float(entry.answer)
        except (TypeError, ValueError):
            return 0.0
