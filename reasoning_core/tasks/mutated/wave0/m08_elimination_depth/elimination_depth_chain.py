import random
import sympy as sp
from dataclasses import dataclass
from typing import List, Dict, Any

from reasoning_core.template import Task, Entry, Config
from reasoning_core.utils import score_scalar


def _rand_nonzero(lo: int, hi: int) -> int:
    if lo > hi:
        lo, hi = hi, lo
    if lo == 0 and hi == 0:
        return 1
    v = random.randint(lo, hi)
    while v == 0:
        v = random.randint(lo, hi)
    return v


def _dependency_depth(eqs: List[sp.Eq], variables: List[sp.Symbol], query: sp.Symbol) -> int:
    """Solver-derived diagnostic: longest dependency-chain length (elimination depth)
    from the queried variable through the equation coupling graph. Two variables are
    coupled when they co-occur in the same equation's support."""
    coupling = {v: set() for v in variables}
    for eq in eqs:
        support = [v for v in variables if eq.has(v)]
        if len(support) >= 2:
            for v in support:
                for u in support:
                    if u is not v:
                        coupling[v].add(u)
    depth = 0
    frontier = {query}
    reached = set(frontier)
    while True:
        nxt = set()
        for v in frontier:
            for u in coupling[v]:
                if u not in reached:
                    nxt.add(u)
        if not nxt:
            break
        reached |= nxt
        frontier = nxt
        depth += 1
    return depth


@dataclass
class EliminationDepthChainCfg(Config):
    total_vars: int = 8
    depth: float = 2.0
    coeff_magnitude: float = 6
    sol_magnitude: float = 5
    max_attempts: int = 200

    def apply_difficulty(self, level):
        self.depth = self.depth + level
        self.coeff_magnitude = self.coeff_magnitude + level


class EliminationDepthChain(Task):
    """Solve for a target variable that lies at a controlled elimination depth
    inside a linear system of approximately fixed dimension; matched nuisance
    equations and variables inflate the system without shortening the chain."""

    config_cls = EliminationDepthChainCfg

    def _try_generate(self, n: int, depth: int):
        variables = list(sp.symbols(f"X1:{n + 1}"))
        variables = random.sample(variables, len(variables))
        chain = variables[: depth + 1]
        query = chain[-1]
        nuis = variables[depth + 1:]

        coeff = int(self.config.coeff_magnitude)
        mag = int(self.config.sol_magnitude)

        sol = {v: _rand_nonzero(-mag, mag) for v in variables}
        eqs = []

        eqs.append(sp.Eq(chain[0], int(sol[chain[0]])))
        for i in range(1, depth + 1):
            a = _rand_nonzero(-coeff, coeff)
            b = _rand_nonzero(-mag, mag)
            eqs.append(sp.Eq(chain[i], a * chain[i - 1] + b))
            sol[chain[i]] = a * sol[chain[i - 1]] + b

        pairs = [nuis[i:i + 2] for i in range(0, len(nuis), 2)]
        for pr in pairs:
            if len(pr) == 2:
                a = _rand_nonzero(-coeff, coeff)
                b = _rand_nonzero(-mag, mag)
                eqs.append(sp.Eq(pr[1], int(sol[pr[1]])))
                eqs.append(sp.Eq(pr[0], a * pr[1] + b))
                sol[pr[0]] = a * sol[pr[1]] + b
            else:
                eqs.append(sp.Eq(pr[0], int(sol[pr[0]])))

        random.shuffle(eqs)

        if not all(sp.simplify(e.lhs.subs(sol) - e.rhs.subs(sol)) == 0 for e in eqs):
            return None

        diagnostic_depth = _dependency_depth(eqs, variables, query)
        if diagnostic_depth < 1:
            return None

        metadata = {
            "equations": [f"{e.lhs} = {e.rhs}" for e in eqs],
            "query_variable": str(query),
            "num_vars": int(n),
            "target_depth": int(depth),
            "diagnostic_depth": int(diagnostic_depth),
            "full_solution_map": {str(k): int(v) for k, v in sol.items()},
        }
        return Entry(metadata=metadata, answer=str(int(sol[query])))

    def generate_entry(self) -> Entry:
        n = int(self.config.total_vars)
        depth = int(round(float(self.config.depth)))
        depth = max(1, min(depth, n - 1))
        for _ in range(self.config.max_attempts):
            entry = self._try_generate(n, depth)
            if entry is not None:
                return entry
        raise RuntimeError(f"Failed to generate a valid problem. Config: {self.config}")

    def render_prompt(self, metadata: dict) -> str:
        eq_block = "\n".join(f"  {e}" for e in metadata["equations"])
        return (
            f"Solve the following system of linear equations for the variable "
            f"'{metadata['query_variable']}'.\n\n"
            f"System:\n{eq_block}\n\n"
            f"The answer is the value of {metadata['query_variable']}."
        )

    def score_answer(self, answer, entry) -> float:
        return score_scalar(answer, entry)


TASK_META = {'parent_source_id': 'f5fdb79ccc11ab61db3cf4aef8d1e1ab292178beaf05cc66da96a0d62626b6d1',
 'idea': 'Test elimination-core depth independently of prompt size.',
 'hypothesis': 'H1',
 'changes': 'Control elimination depth while adding matched nuisance equations '
            'and variables.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3396779050,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
