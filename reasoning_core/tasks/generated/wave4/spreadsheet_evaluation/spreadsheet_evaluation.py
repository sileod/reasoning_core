import random
import re
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload
from reasoning_core.template import stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add spreadsheet formula evaluation with references and dependency '
         'cycles.',
 'hypothesis': 'S44',
 'changes': 'Ask for the value of a named cell, or for the cells that form a '
            'circular reference.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2063019764,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def cell_name(r, c):
    return f"{chr(ord('A') + c)}{r + 1}"


@dataclass
class SpreadsheetEvaluationConfig(Config):
    rows: int = 2
    cols: int = 2
    max_int: int = 20
    max_depth: int = 1
    cyclic_prob: float = 0.2

    def apply_difficulty(self, level):
        self.rows = sround(self.rows + level)
        self.cols = sround(self.cols + level)
        self.max_int = sround(self.max_int + level * 15)
        self.max_depth = sround(self.max_depth + level // 2)
        self.cyclic_prob = 0.1 + 0.02 * level


def _grade_range(rng, grid_rows, grid_cols, depth_limit):
    """Return a random 1x1..depth_limit sized rectangle fitting in the grid."""
    h = rng.randint(1, min(depth_limit, grid_rows))
    w = rng.randint(1, min(depth_limit, grid_cols))
    tr = rng.randint(0, grid_rows - h)
    tc = rng.randint(0, grid_cols - w)
    cells = []
    for rr in range(tr, tr + h):
        for cc in range(tc, tc + w):
            cells.append(cell_name(rr, cc))
    return cells


def _choose_formula(rng, refs, max_depth, max_int, grid_rows, grid_cols, ref_cell):
    """Build a formula expression over given reference cell names."""
    if max_depth <= 0 or not refs:
        return str(rng.randint(1, max_int))
    kind = rng.random()
    if kind < 0.5:
        # basic binary op over two refs / values
        a = rng.choice(refs) if refs and rng.random() < 0.8 else str(rng.randint(1, max_int))
        b = rng.choice(refs) if refs and rng.random() < 0.8 else str(rng.randint(1, max_int))
        op = rng.choice(['+', '-', '*'])
        return f"{a} {op} {b}"
    else:
        # SUM over a rectangular range
        cells = _grade_range(rng, grid_rows, grid_cols, max_depth)
        # ensure at least one ref participates sometimes
        return "SUM(" + ",".join(cells) + ")"


def _evaluate(formula, values, grid_rows, grid_cols):
    """Evaluate a formula given a values dict for named cells. Returns int or None on any ref to unknown sym."""
    nums = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y}
    f = formula.strip()
    m = re.fullmatch(r"SUM\((.*)\)", f)
    if m:
        parts = [p.strip() for p in m.group(1).split(',')]
        total = 0
        for p in parts:
            if p not in values:
                return None
            total += values[p]
        return total
    m = re.fullmatch(r"(\S+)\s+([+\-*])\s+(\S+)", f)
    if m:
        a = m.group(1)
        b = m.group(3)
        va = values.get(a) if a in values else (int(a) if a.isdigit() else None)
        vb = values.get(b) if b in values else (int(b) if b.isdigit() else None)
        if va is None or vb is None:
            return None
        return nums[m.group(2)](va, vb)
    if f.isdigit():
        return int(f)
    if f in values:
        return values[f]
    return None


class SpreadsheetEvaluation(Task):
    config_cls = SpreadsheetEvaluationConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        num_cells = cfg.rows * cfg.cols
        all_names = [cell_name(r, c) for r in range(cfg.rows) for c in range(cfg.cols)]

        # decide cyclic or not
        cyclic = random.random() < cfg.cyclic_prob

        formulas = {}
        # We need a target cell. For acyclic sheets, choose a target with a
        # well-defined value. For cyclic sheets, we will build a cycle and ask
        # for the cycle members.
        target = None
        order = None
        cycle_cells = None

        # Try to build a valid sheet within bounded attempts.
        for _attempt in range(200):
            formulas = {}
            for name in all_names:
                formulas[name] = str(random.randint(1, cfg.max_int))

            refs_pool = [n for n in all_names]

            if cyclic:
                # pick a 2-3 cell cycle
                cyc_len = random.randint(2, min(3, num_cells))
                cyc = random.sample(all_names, cyc_len)
                # each cycle cell directly references the next one, forming a
                # guaranteed cycle.
                for i, name in enumerate(cyc):
                    nxt = cyc[(i + 1) % cyc_len]
                    n0 = random.randint(1, cfg.max_int)
                    if random.random() < 0.5:
                        formulas[name] = f"{nxt} + {n0}"
                    else:
                        formulas[name] = f"{n0} + {nxt}"
                cycle_cells = sorted(cyc)
            else:
                # Build refs for other cells: each non-target cell may be a
                # simple formula over a ref known to be a literal (to stay acyclic),
                # or just literal.
                # target is the last one; make it reference others.
                target = all_names[-1]
                # ensure target is a formula referencing at least one earlier cell (acyclic by construction using indices)
                earlier = all_names[:-1]
                # build a DAG: cell i may reference only cells with smaller "dependency order"
                # We'll just track a "known" literal set.
                known_literals = set()
                for name in all_names:
                    known_literals.add(name)

                # Simpler acyclic scheme: build values in dependency order where
                # each formula only references the immediately preceding literal
                # cells. We'll assign each non-target cell to reference a random
                # earlier cell that is itself a literal - build left to right so
                # dependencies always go leftward -> acyclic.
                cell_list = []
                for i, name in enumerate(all_names):
                    if i == 0:
                        formulas[name] = str(random.randint(1, cfg.max_int))
                    else:
                        # reference a random earlier cell, possibly via SUM over
                        # an earlier-only range, wrapping in a formula
                        ref = random.choice(all_names[:i])
                        depth = random.randint(1, max(1, cfg.max_depth))
                        if random.random() < 0.35:
                            # SUM over earlier cells
                            pool = all_names[:i]
                            h = random.randint(1, min(depth, cfg.rows))
                            w = random.randint(1, min(depth, cfg.cols))
                            sel = sorted(random.sample(pool, min(h * w, len(pool))))
                            formulas[name] = "SUM(" + ",".join(sel) + ")"
                        else:
                            # binary op over an earlier ref and/or a literal
                            a = ref
                            b = (str(random.randint(1, cfg.max_int))
                                 if random.random() < 0.5 else
                                 random.choice(all_names[:i]))
                            op = random.choice(['+', '-', '*'])
                            if random.random() < 0.5:
                                a, b = b, a
                            formulas[name] = f"{a} {op} {b}"
                target = all_names[-1]

                # evaluate to check well-formed & finite
                values = {}
                ok = True
                for name in all_names:
                    v = _evaluate(formulas[name], values, cfg.rows, cfg.cols)
                    if v is None:
                        ok = False
                        break
                    values[name] = v
                if not ok:
                    continue
                order = all_names[:]
                break

            if cyclic:
                break

        if cyclic:
            answer = ",".join(cycle_cells)
            payload = {"grid": {"rows": cfg.rows, "cols": cfg.cols},
                       "formulas": formulas}
            metadata = edict({"formulas": formulas, "cycles": cycle_cells,
                              "rows": cfg.rows, "cols": cfg.cols,
                              "question": "Which cells form a circular reference?"})
            metadata.payload = {"formulas": formulas,
                                "rows": cfg.rows, "cols": cfg.cols}
            prompt_extra = "Which cells form a circular reference?"
        else:
            # evaluate target
            values = {}
            for name in order:
                v = _evaluate(formulas[name], values, cfg.rows, cfg.cols)
                values[name] = v
            answer = str(values[target])
            metadata = edict({"formulas": formulas, "target": target,
                              "rows": cfg.rows, "cols": cfg.cols,
                              "question": f"What is the value of {target}?"})
            metadata.payload = {"formulas": formulas,
                                "rows": cfg.rows, "cols": cfg.cols}
            prompt_extra = f""

        metadata.cyclic = cyclic
        metadata.payload["cyclic"] = cyclic
        if cyclic:
            metadata.payload["question"] = f"Which cells form a circular reference?"
        else:
            metadata.payload["question"] = f"What is the value of {target}?"

        # compute answer separately below to ensure scoring correctness
        if cyclic:
            final_answer = ",".join(cycle_cells)
        else:
            final_answer = str(values[target])

        metadata.cot = ""
        return Entry(metadata=metadata, answer=final_answer)

    def render_prompt(self, metadata):
        rows = metadata.rows
        cols = metadata.cols
        lines = []
        lines.append(f"A spreadsheet has {rows} rows and {cols} columns.")
        for name in sorted(metadata.formulas):
            lines.append(f"{name} = {metadata.formulas[name]}")
        lines.append("")
        lines.append(metadata.question)
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        ground = entry.answer
        if answer is None:
            return 0.0
        s = str(answer).strip()
        g = str(ground).strip()
        if s == g:
            return 1.0
        return 0.0
