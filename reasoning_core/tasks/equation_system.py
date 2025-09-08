import random
import sympy as sp
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from reasoning_core.template import Task, Problem, Config
from reasoning_core.utils import score_scalar

@dataclass
class EquationSystemCfg(Config):
    num_vars: int = 3
    obfuscation_steps: int = 1
    sol_magnitude: int = 30
    coeff_magnitude: int = 4
    max_generation_attempts: int = 200
    p_inconsistent: float = 0.10
    p_underdetermined: float = 0.10
    p_shortcut: float = 0.10

    def update(self, c):
        self.num_vars += c
        self.obfuscation_steps += c

def randint_nonzero(lo: int, hi: int) -> int:
    if lo > hi: lo, hi = hi, lo
    if lo == 0 and hi == 0: return 1
    val = random.randint(lo, hi)
    while val == 0: val = random.randint(lo, hi)
    return val

def _verify_system(equations: List[sp.Eq], variables: List[sp.Symbol]) -> Dict[str, Any]:
    """
    Robustly verifies system properties and returns the solution set for inspection.
    """
    try:
        solution_set = sp.nonlinsolve([eq.lhs - eq.rhs for eq in equations], variables)
        if solution_set == sp.EmptySet:
            return {'kind': 'inconsistent'}
        
        first_sol = next(iter(solution_set))
        if any(s.free_symbols for s in first_sol):
            return {'kind': 'underdetermined', 'solutions': solution_set}
        
        return {'kind': 'unique', 'solutions': solution_set}
    except Exception:
        return {'kind': 'error'}

class EquationSystem(Task):
    def __init__(self, config=EquationSystemCfg()):
        super().__init__(config=config)

    def _generate_base_system(self) -> Tuple[List[sp.Eq], List[sp.Symbol], Dict[sp.Symbol, int]]:
        """Generates a unique system by construction."""
        cfg = self.config
        # Capture dimension once to keep all arrays/loops in sync
        n = int(cfg.num_vars)
        if n < 2:
            return [], [], {}

        variables = list(sp.symbols(f'X1:{n + 1}'))
        sol_map = {v: randint_nonzero(-cfg.sol_magnitude, cfg.sol_magnitude) for v in variables}
        base_exprs = [v - sol_map[v] for v in variables]
        
        C = [[int(i == j) for j in range(n)] for i in range(n)]
        for _ in range(n * cfg.obfuscation_steps):
            i, j = random.sample(range(n), 2)
            k = randint_nonzero(-cfg.coeff_magnitude // 2, cfg.coeff_magnitude // 2)
            for col in range(n):
                C[i][col] += k * C[j][col]
        
        if random.random() < cfg.p_shortcut:
            row_to_simplify = random.randrange(n)
            col_to_keep = random.randrange(n)
            C[row_to_simplify] = [int(j == col_to_keep) for j in range(n)]

        mixed_exprs = [sp.expand(sum(C[i][j] * base_exprs[j] for j in range(n))) for i in range(n)]
        return [sp.Eq(expr, 0) for expr in mixed_exprs], variables, sol_map

    def generate(self) -> Problem:
        for _ in range(self.config.max_generation_attempts):
            eqs, variables, sol_map = self._generate_base_system()
            if not eqs: continue

            # Probabilistically modify the base system
            rand_val = random.random()
            was_modified = False
            if rand_val < self.config.p_inconsistent:
                i, j = random.sample(range(len(eqs)), 2)
                eqs[j] = sp.Eq(eqs[i].lhs, eqs[i].rhs + randint_nonzero(-10, 10))
                was_modified = True
            elif rand_val < self.config.p_inconsistent + self.config.p_underdetermined:
                eqs.pop(random.randrange(len(eqs)))
                was_modified = True
            
            verification = _verify_system(eqs, variables)
            case = verification['kind']
            if case == 'error': continue

            query_var = random.choice(variables)
            answer = None

            if case == 'unique':
                if was_modified: continue
                answer = sol_map[query_var]
            elif case == 'inconsistent':
                answer = "No solution"
            elif case == 'underdetermined':
                var_idx = variables.index(query_var)
                sol_expr = next(iter(verification['solutions']))[var_idx]
                if not sol_expr.free_symbols:
                    answer = sp.N(sol_expr)
                    case = "underdetermined_but_unique_var"
                else:
                    answer = "Multiple solutions"

            if answer is None: continue

            metadata = {
                "equations": [f"{eq.lhs} = {eq.rhs}" for eq in eqs],
                "query_variable": str(query_var),
                "full_solution_map": {str(k): int(v) for k, v in sol_map.items()} if not was_modified else None,
                "case": case
            }
            return Problem(metadata=metadata, answer=str(answer))

        raise RuntimeError(f"Failed to generate a valid problem. Config: {self.config}")

    def prompt(self, metadata: dict) -> str:
        eq_block = "\n".join([f"  {eq_str}" for eq_str in metadata['equations']])
        return (f"Solve the following system of equations for the variable '{metadata['query_variable']}'.\n\n"
                f"System:\n{eq_block}\n\n"
                f"Return the numerical value for {metadata['query_variable']}. If a unique numerical solution does not exist, "
                "return either 'No solution' or 'Multiple solutions'.")


    def score_answer(self, answer, entry) -> float:
        normalize = lambda text: str(text).split('=')[-1].lower().strip().replace('_', ' ').replace('-', ' ')
        a = normalize(answer)
        if "solution" in a:
            return float(a==normalize(entry.answer))
        if "solution" in entry.answer.lower():
            return 0.0
        return score_scalar(answer, entry)