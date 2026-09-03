from dataclasses import dataclass
import math
import random

from reasoning_core.template import Task, Entry, Config, edict


def _merge_pair(a, m, b, n):
    """Combine x ≡ a (mod m) and x ≡ b (mod n). Returns (x, lcm) or None if
    inconsistent, using the extended Euclidean / modular inverse."""
    g = math.gcd(m, n)
    if (b - a) % g != 0:
        return None
    lcm = m // g * n
    m1, n1 = m // g, n // g
    inv = pow(m1, -1, n1)
    t = ((b - a) // g) * inv % n1
    x = (a + m * t) % lcm
    return x, lcm


def merge_all(congruences):
    """Solve all (m, r) congruences jointly. Returns (x, lcm) or None."""
    if not congruences:
        return None
    a = congruences[0][1]
    m = congruences[0][0]
    for (n, b) in congruences[1:]:
        res = _merge_pair(a, m, b, n)
        if res is None:
            return None
        a, m = res
    return a, m


def canonical_answer(congruences):
    """Smallest non-negative solution as a string, or 'none'."""
    res = merge_all(congruences)
    if res is None:
        return "none"
    return str(res[0])


@dataclass
class CongruenceSystemConfig(Config):
    n_min: int = 3
    n_max: int = 3
    base_bits: int = 7

    def apply_difficulty(self, level):
        self.n_min = min(6, 3 + level)
        self.n_max = min(6, 3 + level)
        self.base_bits = 6 + level


class CongruenceSystem(Task):
    config_cls = CongruenceSystemConfig

    def generate_entry(self):
        cfg = self.config
        n = random.randint(cfg.n_min, cfg.n_max)

        # build a jointly consistent baseline with overlapping moduli
        congruences, lcm = self._build_consistent(n)

        inconsistent = random.random() < 0.25
        if inconsistent:
            congruences = self._make_inconsistent(congruences, n)

        answer = canonical_answer(congruences)
        if answer != "none":
            # answer must reach six digits at the top level so it cannot be guessed
            padded = int(answer)
            if padded < 10 ** 6:
                # lift to a large representative of the same residue class, with
                # a spread that grows with difficulty so answers stay varied
                span = cfg.base_bits
                k = (10 ** 6 - padded) // lcm + random.randrange(0, span * 3)
                padded += k * lcm
            answer = str(padded)
            congruences = [(m, padded % m) for (m, r) in congruences]

        items = []
        for m, r in congruences:
            items.append({
                "m": int(m),
                "r": int(r),
                "text": "a number that leaves remainder %d when divided by %d"
                        % (int(r % m), int(m)),
            })

        metadata = edict({
            "congruences": items,
            "system": [[int(m), int(r)] for (m, r) in congruences],
            "answer": answer,
        })
        metadata.payload = {"givens": [it["text"] for it in items]}
        return Entry(metadata=metadata, answer=answer)

    def _build_consistent(self, n):
        cfg = self.config
        while True:
            base = random.randrange(2 ** (cfg.base_bits // 2), 2 ** (cfg.base_bits // 2 + 3) + 1)
            moduli = [base * random.randrange(1, 4) for _ in range(n)]
            pivot = random.randrange(0, max(moduli) + 1)
            congruences = [(m, pivot % m) for m in moduli]
            res = merge_all(congruences)
            if res is not None:
                return congruences, res[1]

    def _make_inconsistent(self, congruences, n):
        # choose a pair that is NOT adjacent in the list
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i > 1]
        i, j = random.choice(pairs)
        mi, ri = congruences[i]
        mj, rj = congruences[j]
        g = math.gcd(mi, mj)
        # nudge ri so that the pair becomes inconsistent modulo g
        new_ri = (ri + 1) % mi
        if (new_ri - rj) % g == 0:
            new_ri = (new_ri + 1) % mi
        congruences[i] = (mi, new_ri)
        # verify inconsistency
        assert merge_all(congruences) is None
        return congruences

    def render_prompt(self, metadata):
        givens = metadata.payload["givens"]
        body = "\n".join(g + "." for g in givens)
        return (
            body
            + "\n\nFind the smallest non-negative integer that satisfies all of the "
            + "conditions above. If no such integer exists, answer exactly \"none\"."
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        if isinstance(answer, str):
            answer = answer.strip()
        return 1.0 if answer == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Add simultaneous congruences with moduli that are not coprime.',
 'hypothesis': 'S51',
 'changes': 'Ask for the smallest non-negative solution, or for a proof that '
            'none exists.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2311544729,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
