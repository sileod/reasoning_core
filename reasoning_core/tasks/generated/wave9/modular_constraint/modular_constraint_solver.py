import math
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload


def _egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _inv_mod(a, m):
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("no inverse")
    return x % m


def _merge(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    if (r2 - r1) % g != 0:
        return None
    lcm = m1 // g * m2
    if m1 == 1:
        return r2 % m2, m2
    if m2 == 1:
        return r1 % m1, m1
    t = ((r2 - r1) // g) % (m2 // g)
    x = (r1 + m1 * t * _inv_mod(m1 // g, m2 // g)) % lcm
    return x, lcm


def _solve_system(congruences):
    x = 0
    m = 1
    for r, mod in congruences:
        merged = _merge(x, m, r, mod)
        if merged is None:
            return None
        x, m = merged
    return x % m, m


def _reduced(r, m):
    return r % m


def _reduce_by_gcd(congruences):
    out = []
    for r, mod in congruences:
        r = _reduced(r, mod)
        if mod == 1:
            continue
        out.append((r, mod))
    return out


def _force_inconsistent(mods):
    """Build a guaranteed-inconsistent congruence set from given moduli.

    Pick two moduli sharing a factor >= 2 and give them incompatible residues
    modulo that common factor.
    """
    n = len(mods)
    i, j, g = -1, -1, 1
    for a in range(n):
        for b in range(a + 1, n):
            gg = math.gcd(mods[a], mods[b])
            if gg > 1:
                i, j, g = a, b, gg
                break
        if i != -1:
            break
    if i == -1:
        i, j = 0, 1 % n
        g = 1
    cong = []
    for k in range(n):
        if k == i:
            r = 0
        elif k == j and g > 1:
            r = 1 % mods[k]
        else:
            r = random.randrange(mods[k])
        cong.append((r, mods[k]))
    return cong


@dataclass
class ModularConstraintConfig(Config):
    n_constraints: int = 3
    max_mod: int = 12
    allow_non_coprime: float = 0.6
    inconsistency_prob: float = 0.15

    def apply_difficulty(self, level):
        self.n_constraints = max(2, 3 + (level + 1) // 2)
        self.max_mod = 8 + 3 * level
        self.allow_non_coprime = min(0.9, 0.4 + 0.1 * level)
        self.inconsistency_prob = 0.15


class ModularConstraint(Task):
    summary = ("Combine modular congruence constraints including non-coprime moduli, "
               "returning inconsistency or the canonical residue and modulus solution.")
    config_cls = ModularConstraintConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_constraints

        while True:
            mods = []
            for _ in range(n):
                mod = random.randint(3, cfg.max_mod)
                mods.append(mod)

            # decide consistency
            inconsistent = random.random() < cfg.inconsistency_prob

            if inconsistent:
                has_factored = any(
                    math.gcd(mods[a], mods[b]) > 1
                    for a in range(n) for b in range(a + 1, n)
                )
                if not has_factored:
                    continue
                for _ in range(500):
                    congruences = []
                    for mod in mods:
                        r = random.randrange(mod)
                        congruences.append((r, mod))
                    if _solve_system(congruences) is None:
                        break
                else:
                    congruences = _force_inconsistent(mods)
                reduced = _reduce_by_gcd(congruences)
                if not reduced:
                    continue
                if _solve_system(reduced) is not None:
                    continue
                answer = "inconsistent"
            else:
                base_r = random.randrange(cfg.max_mod)
                congruences = []
                for mod in mods:
                    r = base_r % mod
                    congruences.append((r, mod))
                reduced = _reduce_by_gcd(congruences)
                sol = _solve_system(reduced)
                # sol must exist; verify by applying
                ok = True
                for r, m in reduced:
                    if sol[0] % m != r % m:
                        ok = False
                if not ok:
                    continue
                x, m = sol
                x = x % m
                answer = f"{x} mod {m}"

            payload = {
                "constraints": [f"x \u2261 {r} (mod {m})" for r, m in reduced],
            }
            metadata = edict({
                "constraints": [[r, m] for r, m in reduced],
                "inconsistent": (answer == "inconsistent"),
            })
            metadata.payload = payload
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = metadata.payload["constraints"]
        body = "\n".join(lines)
        return (
            f"Combine these modular congruence constraints into a single solution:\n"
            f"{body}\n\n"
            f"If the system has a solution, give it as the least non-negative residue "
            f"x and its combined modulus in the form \"x mod m\". "
            f"If the constraints cannot all hold simultaneously, answer exactly "
            f"\"inconsistent\". "
            f"For example, for the constraints x \u2261 3 (mod 4) and x \u2261 1 (mod 2), "
            f"the answer is \"3 mod 4\"."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        answer = answer.strip()
        if not answer:
            return 0.0
        if entry.metadata.inconsistent:
            return 1.0 if answer == "inconsistent" else 0.0
        gold = entry.answer
        return 1.0 if answer == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'modular_constraint_solver (draw 1 of 1)',
 'hypothesis': 'HV-061',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/modular_constraint_solver',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3591494931,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
