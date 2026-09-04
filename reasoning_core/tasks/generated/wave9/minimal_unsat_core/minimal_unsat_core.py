from dataclasses import dataclass
import random
import ast

from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.template import stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'minimal_unsat_core (draw 1 of 1)',
 'hypothesis': 'HV-051',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/minimal_unsat_core',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1787056888,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def neg(v):
    return 1 - v


def sat(clauses, nvars):
    for mask in range(1 << nvars):
        assign = {i: (mask >> i) & 1 for i in range(nvars)}
        if all(c.satisfied(assign) for c in clauses):
            return True
    return False


class Clause:
    def __init__(self, lits):
        self.lits = list(lits)

    def satisfied(self, assign):
        for var in self.lits:
            b, s = var
            if isinstance(b, bool):
                v = 1 if b else 0
            else:
                v = assign[b]
            if s is False:
                v = neg(v)
            if v == 1:
                return True
        return False

    def render(self):
        parts = []
        for var in self.lits:
            b, s = var
            if isinstance(b, bool):
                lit = "T" if b else "F"
            else:
                lit = f"x{b}"
            if s is False:
                lit = "~" + lit
            parts.append(lit)
        return "(" + " or ".join(parts) + ")"


def make_clause(n_vars, bool_frac, rng):
    if bool_frac > 0 and rng.random() < bool_frac:
        return Clause([(bool(rng.random() < 0.5), True)])
    width = min(n_vars, rng.choice([1, 2, 2, 2, 3]))
    vars_ = rng.sample(range(n_vars), k=width)
    signed = [rng.random() < 0.5 for _ in range(width)]
    return Clause([(b, s) for b, s in zip(vars_, signed)])


def nvars_of(clauses):
    m = -1
    for c in clauses:
        for b, _ in c.lits:
            if not isinstance(b, bool) and b > m:
                m = b
    return m + 1


@dataclass
class MinimalUnsatCoreConfig(Config):
    n_constraints: int = 7
    n_vars: int = 3

    def apply_difficulty(self, level):
        self.n_constraints = sround(5 + level * 1.3)
        self.n_vars = sround(2 + level * 1.0)
        self.bool_frac = min(0.35, 0.05 + level * 0.05)


class MinimalUnsatCore(Task):
    summary = ("Find the lexicographically canonical subset-minimal inconsistent subset "
               "of small Boolean or finite-domain constraints with varied clause widths "
               "(unit, binary, ternary), sign patterns and occasional constant literals.")
    config_cls = MinimalUnsatCoreConfig

    def generate_entry(self):
        cfg = self.config
        n_constraints = int(cfg.n_constraints)
        n_vars = max(2, min(12, int(cfg.n_vars)))
        bool_frac = float(getattr(cfg, 'bool_frac', 0.05))

        while True:
            clauses = [make_clause(n_vars, bool_frac, random)
                       for _ in range(n_constraints)]
            cores = self._minimal_cores(clauses)
            if cores:
                break

        smallest = min(len(c) for c in cores)
        min_cores = [c for c in cores if len(c) == smallest]
        lex_min = min(min_cores, key=lambda c: tuple(sorted(c)))

        metadata = edict({
            "n_vars": n_vars,
            "core": sorted(int(i) for i in lex_min),
        })
        metadata.payload = {
            "Variables": [f"x{i}" for i in range(n_vars)],
            "Constraints": [c.render() for c in clauses],
        }
        answer = ",".join(str(int(i)) for i in sorted(lex_min))
        return Entry(metadata=metadata, answer=answer)

    def _subsets(self, n, size):
        if size > n:
            return []
        result = []
        idxs = list(range(size))
        while True:
            result.append(list(idxs))
            i = size - 1
            while i >= 0 and idxs[i] == n - size + i:
                i -= 1
            if i < 0:
                break
            idxs[i] += 1
            for j in range(i + 1, size):
                idxs[j] = idxs[j - 1] + 1
        return result

    def _minimal_cores(self, clauses):
        n = len(clauses)
        nvars = nvars_of(clauses)
        cores = []
        smallest_so_far = None
        subs = []
        if n >= 2:
            subs = self._subsets(n, 2) + self._subsets(n, 3)
        subs.sort()
        for subset in subs:
            sub = [clauses[i] for i in subset]
            if not sat(sub, nvars):
                if self._is_minimal(clauses, subset, nvars):
                    cores.append(subset)
                    if smallest_so_far is None or len(subset) < smallest_so_far:
                        smallest_so_far = len(subset)
        if not cores:
            for size in range(2, n + 1):
                for subset in self._subsets(n, size):
                    sub = [clauses[i] for i in subset]
                    if not sat(sub, nvars):
                        if self._is_minimal(clauses, subset, nvars):
                            cores.append(subset)
        return cores

    def _is_minimal(self, clauses, subset, nvars):
        for k in range(len(subset)):
            rest = [subset[j] for j in range(len(subset)) if j != k]
            if not sat([clauses[i] for i in rest], nvars):
                return False
        return True

    def render_prompt(self, metadata):
        var_line = "Variables: " + ", ".join(f"x{i} in {{0,1}}" for i in range(metadata.n_vars))
        cons_lines = "\n".join(f"  {i}: {c}" for i, c in enumerate(metadata.payload["Constraints"]))
        return (f"{var_line}\nConstraints:\n{cons_lines}\n\n"
                f"The whole set of constraints is unsatisfiable. Find a lexicographically canonical "
                f"subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is "
                f"unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such "
                f"subsets, report one with the minimum number of constraints; if several tie, report the "
                f"one whose sorted constraint indices form the lexicographically smallest tuple.\n"
                f"Answer with the sorted constraint indices as a comma-separated list, e.g. \"0,2\" or \"1\".")

    def score_answer(self, answer, entry):
        try:
            parsed = ast.literal_eval(answer)
        except (SyntaxError, ValueError):
            return 0.0
        if isinstance(parsed, (bool, int, float)):
            parsed = [parsed]
        elif not isinstance(parsed, (list, tuple)):
            return 0.0
        try:
            parsed = sorted(int(x) for x in parsed)
        except (TypeError, ValueError):
            return 0.0
        gold = sorted(int(i) for i in entry.metadata.core)
        return 1.0 if parsed == gold else 0.0
