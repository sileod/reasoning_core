import random

from sympy import ilcm
from sympy.ntheory.modular import solve_congruence

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add modular linear-congruence reasoning.',
 'hypothesis': 'N2',
 'changes': 'Implement canonical residue, no-solution, and multiple-solution '
            'queries for modular systems.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 533370632,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class ModularCongruenceSystemConfig(Config):
    n_cong: int = 3
    max_mod: int = 11
    no_sol_prob: float = 0.25

    def apply_difficulty(self, level):
        self.n_cong = sround(self.n_cong + level)
        self.max_mod = sround(self.max_mod + 2 * level)
        self.no_sol_prob = min(0.35, 0.25 + 0.02 * level)


def _consistent_system(n, max_mod):
    for _ in range(2000):
        mods = [random.randint(2, max_mod) for _ in range(n)]
        lcm = ilcm(*mods)
        t = random.randint(0, lcm - 1)
        residues = [int(t % m) for m in mods]
        if solve_congruence(*zip(residues, mods)) is not None:
            return mods, residues, int(t)
    raise RuntimeError("could not build consistent congruence system")


def _inconsistent_system(n, max_mod):
    for _ in range(2000):
        d = random.randint(2, max_mod)
        m1 = d * random.randint(1, max_mod // d)
        m2 = d * random.randint(1, max_mod // d)
        mods = [m1, m2]
        while len(mods) < n:
            mods.append(random.randint(2, max_mod))
        r1 = random.randint(0, m1 - 1)
        r2 = random.randint(0, m2 - 1)
        guard = 0
        while (r1 - r2) % d == 0 and guard < 1000:
            r2 = random.randint(0, m2 - 1)
            guard += 1
        residues = [r1, r2]
        for m in mods[2:]:
            residues.append(random.randint(0, m - 1))
        if solve_congruence(*zip(residues, mods)) is None:
            return mods, residues
    raise RuntimeError("could not build inconsistent congruence system")


class ModularCongruenceSystem(Task):
    config_cls = ModularCongruenceSystemConfig

    def generate_entry(self):
        n = self.config.n_cong
        max_mod = self.config.max_mod
        if random.random() < self.config.no_sol_prob:
            mods, residues = _inconsistent_system(n, max_mod)
            has_solution = False
            canonical = None
        else:
            mods, residues, t = _consistent_system(n, max_mod)
            has_solution = True
            canonical = t
        congruences = [f"x ≡ {int(r)} (mod {int(m)})" for r, m in zip(residues, mods)]
        metadata = edict({
            'n_cong': int(n),
            'mods': [int(m) for m in mods],
            'residues': [int(r) for r in residues],
            'has_solution': has_solution,
            'canonical': canonical,
            'lcm': int(ilcm(*mods)),
        })
        metadata.payload = {'congruences': congruences}
        if has_solution:
            answer = str(int(canonical))
        else:
            answer = "NONE"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            f"Determine whether there is an integer x satisfying every congruence "
            f"in the system above. If yes, give the smallest non-negative integer "
            f"solution (the canonical residue modulo the least common modulus of the "
            f"moduli). If no integer satisfies all congruences, write NONE. "
            f"The answer is a single integer, or the word NONE."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        text = str(answer).strip()
        if entry.metadata.has_solution:
            if text == "NONE":
                return 0.0
            try:
                return 1.0 if int(text) == int(entry.metadata.canonical) else 0.0
            except ValueError:
                return 0.0
        else:
            return 1.0 if text.upper() == "NONE" else 0.0
