from dataclasses import dataclass
import ast
import json
import math
import random
from typing import Optional

from reasoning_core.template import Task, Problem, Config, edict
from z3 import Distinct, Int, Optimize, Or, Solver, Sum, sat


@dataclass
class ConstraintSatisfactionConfig(Config):
    n_vars: int = 2
    max_domain: int = 2
    n_constraints: int = 3
    coef_bound: int = 3
    unsat_prob: float = 0.15
    max_tries: int = 64

    # Structure: "random" | "graph" | "grid" | "clustered" | "any"
    structure_mode: str = "any"
    max_arity: int = 3

    # Solve mode: "all" (enumerate all) or "min" (lex-smallest). 
    # If "all" overflows `max_solutions`, it automatically falls back to "min".
    solve_mode: str = "all"
    max_solutions: Optional[int] = 256

    edge_prob: float = 0.25
    n_clusters: int = 3
    p_in: float = 0.6
    p_out: float = 0.1
    grid_width: Optional[int] = None

    def update(self, c=1):
        self.n_vars += 0.6 * c
        self.max_domain += 0.4 * c
        self.n_constraints += 1.1 * c
        self.coef_bound += 0.3 * c
        self.max_arity = min(4, self.max_arity + int(c >= 3))


CSPConfig = ConstraintSatisfactionConfig


class ConstraintSatisfaction(Task):

    def __init__(self, config=None):
        super().__init__(config=config or CSPConfig())

    def _rng(self):
        return random.Random(self.config.seed)

    def _build_neighbors(self, rng, n, mode):
        nbrs = [set() for _ in range(n)]

        def link(i, j):
            if i != j:
                nbrs[i].add(j); nbrs[j].add(i)

        if mode == "random":
            for i in range(n): nbrs[i] = set(range(n)) - {i}
        elif mode == "graph":
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < self.config.edge_prob: link(i, j)
        elif mode == "grid":
            w = self.config.grid_width or max(1, round(math.sqrt(n)))
            h = (n + w - 1) // w
            for r in range(h):
                for c in range(w):
                    i = r * w + c
                    if i < n:
                        for dr, dc in ((1, 0), (0, 1)):
                            j = (r + dr) * w + (c + dc)
                            if r + dr < h and c + dc < w and j < n: link(i, j)
        elif mode == "clustered":
            g = max(1, min(self.config.n_clusters, n))
            cluster_of = [rng.randrange(g) for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    p = self.config.p_in if cluster_of[i] == cluster_of[j] else self.config.p_out
                    if rng.random() < p: link(i, j)

        for i in range(n):
            if not nbrs[i]: link(i, rng.choice([x for x in range(n) if x != i]))
        return nbrs

    def _sample_scope(self, rng, neighbors, n, mode):
        k = rng.randint(1, max(1, min(self.config.max_arity, n)))
        if mode == "random" or k == 1:
            return sorted(rng.sample(range(n), k))

        seed = rng.randrange(n)
        scope, frontier = {seed}, set(neighbors[seed])
        while len(scope) < k and frontier:
            j = rng.choice(tuple(frontier))
            scope.add(j)
            frontier.update(neighbors[j])
            frontier.difference_update(scope)

        if len(scope) < k:
            rest = [j for j in range(n) if j not in scope]
            if rest: scope.update(rng.sample(rest, min(k - len(scope), len(rest))))
        return sorted(scope)

    def _sample_constraint(self, rng, idx, witness, coef_bound):
        kind = rng.choices(["lin", "mod", "alldiff"], weights=[6, 2, 2] if len(idx) >= 2 else [7, 3, 0])[0]

        if kind == "lin":
            coeffs = [rng.choice([x for x in range(-coef_bound, coef_bound + 1) if x != 0]) for _ in idx]
            val = sum(a * witness[i] for a, i in zip(coeffs, idx))
            op = rng.choices(["==", "!=", "<=", ">="], weights=[3, 2, 3, 3])[0]
            if op == "==": rhs = val
            elif op == "!=": rhs = val + rng.choice([x for x in range(-(coef_bound + 2), coef_bound + 3) if x != 0])
            elif op == "<=": rhs = val + rng.randint(0, coef_bound + 1)
            else: rhs = val - rng.randint(0, coef_bound + 1)
            return {"type": "lin", "idx": idx, "coeffs": coeffs, "op": op, "rhs": rhs}

        if kind == "mod":
            coeffs = [rng.randint(1, coef_bound) for _ in idx]
            mod = rng.randint(2, max(2, coef_bound + 2))
            rem = sum(a * witness[i] for a, i in zip(coeffs, idx)) % mod
            return {"type": "mod", "idx": idx, "coeffs": coeffs, "mod": mod, "rem": rem}

        if len(set(witness[i] for i in idx)) != len(idx): return None
        return {"type": "alldiff", "idx": idx}

    def _constraint_text(self, c):
        if c["type"] in ("lin", "mod"):
            parts = [(f"x{i}" if a == 1 else f"-x{i}" if a == -1 else f"{a}*x{i}") for a, i in zip(c['coeffs'], c['idx'])]
            expr = " + ".join(parts).replace("+ -", "- ")
            if c["type"] == "lin": return f"{expr} {c['op']} {c['rhs']}"
            return f"({expr}) % {c['mod']} == {c['rem']}"
        return f"AllDifferent({', '.join(f'x{i}' for i in c['idx'])})"

    def _add_base(self, solver, xs, domains, constraints):
        for x, ub in zip(xs, domains): solver.add(x >= 0, x <= ub)
        for c in constraints:
            if c["type"] == "lin":
                expr = Sum([a * xs[i] for a, i in zip(c["coeffs"], c["idx"])])
                if c["op"] == "==": solver.add(expr == c["rhs"])
                elif c["op"] == "!=": solver.add(expr != c["rhs"])
                elif c["op"] == "<=": solver.add(expr <= c["rhs"])
                else: solver.add(expr >= c["rhs"])
            elif c["type"] == "mod":
                solver.add(Sum([a * xs[i] for a, i in zip(c["coeffs"], c["idx"])]) % c["mod"] == c["rem"])
            elif c["type"] == "alldiff":
                solver.add(Distinct(*[xs[i] for i in c["idx"]]))

    def _solve_min(self, domains, constraints):
        xs = [Int(f"x{i}") for i in range(len(domains))]
        opt = Optimize()
        opt.set(priority="lex")
        self._add_base(opt, xs, domains, constraints)
        for x in xs: opt.minimize(x)
        if opt.check() != sat: return None
        return [opt.model().eval(x, model_completion=True).as_long() for x in xs]

    def _solve_all(self, domains, constraints):
        xs = [Int(f"x{i}") for i in range(len(domains))]
        solver = Solver()
        self._add_base(solver, xs, domains, constraints)
        solutions, cap = [], self.config.max_solutions
        while solver.check() == sat:
            sol = [solver.model().eval(x, model_completion=True).as_long() for x in xs]
            solutions.append(sol)
            if cap and len(solutions) > cap: return None, True
            solver.add(Or(*[x != v for x, v in zip(xs, sol)]))
        return sorted(solutions) if solutions else None, False

    def generate(self):
        rng = self._rng()
        n, max_dom, n_cons = max(2, self.config.n_vars), max(2, self.config.max_domain), max(1, self.config.n_constraints)

        for _ in range(self.config.max_tries):
            # Dynamic structure fallback for pure randomness per instance
            mode = rng.choice(["random", "graph", "grid", "clustered"]) if self.config.structure_mode == "any" else self.config.structure_mode
            neighbors = self._build_neighbors(rng, n, mode)
            domains = [rng.randint(1, max_dom) for _ in range(n)]
            witness = [rng.randint(0, ub) for ub in domains]

            constraints, seen, attempts = [], set(), 0
            while len(constraints) < n_cons and attempts < max(16, 12 * n_cons):
                attempts += 1
                idx = self._sample_scope(rng, neighbors, n, mode)
                if c := self._sample_constraint(rng, idx, witness, self.config.coef_bound):
                    if (key := json.dumps(c, sort_keys=True)) not in seen:
                        seen.add(key); constraints.append(c)

            if len(constraints) < max(1, n_cons // 2): continue

            if rng.random() < self.config.unsat_prob:
                i = rng.randrange(n)
                a = rng.randint(0, domains[i])
                b = rng.choice([x for x in range(domains[i] + 1) if x != a] or [a])
                constraints += [
                    {"type": "lin", "idx": [i], "coeffs": [1], "op": "==", "rhs": a},
                    {"type": "lin", "idx": [i], "coeffs": [1], "op": "==", "rhs": b},
                ]

            solve_mode = self.config.solve_mode.lower()
            if solve_mode in ("all", "lex_all"):
                solution, overflow = self._solve_all(domains, constraints)
                # Fallback to Min if "all" hits bounds
                if overflow:
                    solve_mode, solution = "min", self._solve_min(domains, constraints)
            else:
                solution = self._solve_min(domains, constraints)

            metadata = edict({
                "domains": domains, "constraints": constraints, "solution": solution,
                "solve_mode": solve_mode, "structure_mode": mode,
                "instance": "Variables/domains:\n" + \
                            "\n".join(f"- 0 <= x{i} <= {ub}" for i, ub in enumerate(domains)) + \
                            "\n\nConstraints:\n" + "\n".join(f"{j+1}. {self._constraint_text(c)}" for j, c in enumerate(constraints))
            })
            return Problem(metadata=metadata, answer="UNSAT" if solution is None else json.dumps(solution))
            
        raise RuntimeError("Failed to generate a CSP instance.")

    def prompt(self, metadata):
        order = ", ".join(f"x{i}" for i in range(len(metadata['domains'])))
        if metadata.get("solve_mode", "min") == "all":
            return f"{metadata['instance']}\n\nEnumerate ALL satisfying assignments in variable order [{order}].\nReturn them as a Python list of lists of ints, sorted lexicographically.\nIf no assignment exists, return UNSAT.\n"
        return f"{metadata['instance']}\n\nReturn the lexicographically smallest satisfying assignment in variable order [{order}] as a Python list of ints.\nIf no assignment exists, return UNSAT.\n"

    def score_answer(self, answer, entry):
        def _parse(s):
            if not isinstance(s, str) or not (s := s.strip()): return None
            if s.upper() == "UNSAT": return "UNSAT"
            try:
                def norm(v): return [norm(u) for u in v] if isinstance(v, (list, tuple)) else v
                x = norm(ast.literal_eval(s))
                if isinstance(x, list) and (all(type(v) is int for v in x) or all(isinstance(r, list) and all(type(v) is int for v in r) for r in x)):
                    return x
            except Exception: pass
            return None

        parsed = _parse(answer)
        metadata = entry.metadata if hasattr(entry, "metadata") else entry["metadata"]
        expected = metadata["solution"]
        if expected is None: return float(parsed == "UNSAT")
        if parsed == "UNSAT" or not isinstance(parsed, list): return 0.0
        
        # Validates proper dimensionality match ("all" == 2D array, "min" == 1D array)
        if metadata.get("solve_mode", "min") == "all":
            return float(parsed == expected and (not parsed or isinstance(parsed[0], list)))
        return float(parsed == expected and (not parsed or not isinstance(parsed[0], list)))


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: 'any' mode (Randomly cycles graphs), default 'all' (Fallbacks to 'min')")
    print("=" * 60)
    for _ in range(4):
        cfg = ConstraintSatisfactionConfig(
            n_vars=4,
            max_domain=3,
            n_constraints=5,
            coef_bound=2,
            structure_mode="any",
            solve_mode="all",
            max_solutions=32,
            seed=random.randint(0, 1000)
        )
        task = ConstraintSatisfaction(config=cfg)
        prob = task.generate()
        
        print(f"\n--- Structure chosen: {prob.metadata.structure_mode} ---")
        print(task.prompt(prob.metadata))
        
        sol = json.loads(prob.answer) if prob.answer != "UNSAT" else "UNSAT"
        count = len(sol) if isinstance(sol, list) and sol and isinstance(sol[0], list) else (0 if sol == "UNSAT" else 1)
        print(f"Answer ({count} solutions, solve_mode used={prob.metadata.solve_mode}):\n{prob.answer}\n")