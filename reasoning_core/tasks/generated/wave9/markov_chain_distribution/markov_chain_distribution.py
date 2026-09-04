from dataclasses import dataclass
import random
from math import gcd
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'markov_chain_distribution (draw 1 of 1)',
 'hypothesis': 'HV-003',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/markov_chain_distribution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2267388306,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class MarkovChainConfig(Config):
    n_states: int = 3
    steps: int = 2

    def apply_difficulty(self, level):
        self.n_states = sround(self.n_states + level)
        self.steps = sround(self.steps + level)


def _parse_prob(answer):
    answer = str(answer).strip()
    if "/" in answer:
        parts = answer.split("/")
        if len(parts) != 2:
            raise ValueError("bad")
        try:
            n = float(parts[0].strip())
            d = float(parts[1].strip())
        except ValueError:
            raise ValueError("bad")
        if d == 0:
            raise ValueError("bad")
        return n / d
    try:
        return float(answer)
    except ValueError:
        raise ValueError("bad")


def _propagate_fraction(int_rows, denom, start, target, steps):
    n = len(int_rows)
    v = [Fraction(0, 1) for _ in range(n)]
    v[start] = Fraction(1, 1)
    for _ in range(steps):
        nv = [Fraction(0, 1) for _ in range(n)]
        for i in range(n):
            if v[i]:
                for j in range(n):
                    if int_rows[i][j]:
                        nv[j] += v[i] * int_rows[i][j] / denom
        v = nv
    return v[target]


def _reduce(num, den):
    g = gcd(num, den)
    return num // g, den // g


class MarkovChainDistribution(Task):
    summary = "Propagate exact probability mass through finite Markov chains for a stated number of steps, returning a queried state or event probability."
    config_cls = MarkovChainConfig
    task_version = 2

    def generate_entry(self):
        n = self.config.n_states
        steps = self.config.steps

        for _ in range(500):
            denom = random.choice([2, 3, 4, 5, 6, 10])
            rows = []
            for _ in range(n):
                entries = [0] * n
                nz = random.randint(1, min(n, denom))
                sel = random.sample(range(n), nz)
                extra = denom - nz
                for s in sel:
                    entries[s] = 1
                for _ in range(extra):
                    entries[random.choice(sel)] += 1
                if sum(entries) != denom:
                    continue
                rows.append(entries)
            if not all(sum(r) == denom for r in rows):
                continue

            start = random.randrange(n)
            target = random.randrange(n)
            if start == target and steps >= 1:
                pass
            prob = _propagate_fraction(rows, denom, start, target, steps)
            if prob <= 0 or prob > 1:
                continue
            break
        else:
            denom = 2
            rows = [[1, 1], [1, 1]]
            start, target, steps = 0, 1, 1
            prob = Fraction(1, 2)

        np_, dp_ = _reduce(prob.numerator, prob.denominator)
        answer = f"{np_}/{dp_}" if dp_ != 1 else f"{np_}"

        matrix = [[r / denom for r in row] for row in rows]

        metadata = edict({
            "states": n,
            "steps": steps,
            "start": start,
            "target": target,
            "matrix": matrix,
            "event_prob": float(prob),
        })
        metadata.payload = {
            "states": list(range(n)),
            "start_state": start,
            "target_state": target,
            "steps": steps,
            "transition_matrix_rows": [[float(c) for c in r] for r in matrix],
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        rows_str = "\n".join(
            "  From state {}: [{}]".format(i, ", ".join(f"{x:g}" for x in r))
            for i, r in enumerate(p["transition_matrix_rows"])
        )
        return (
            f"A Markov chain has states {', '.join(str(s) for s in p['states'])}. "
            f"Its one-step transition matrix is:\n{rows_str}\n\n"
            f"The chain is in state {p['start_state']} at time 0. After exactly "
            f"{p['steps']} steps, what is the probability of being in state "
            f"{p['target_state']}?\n\n"
            f"The answer is the exact probability as a reduced fraction a/b (write a when b is 1)."
        )

    def score_answer(self, answer, entry):
        try:
            val = _parse_prob(answer)
        except ValueError:
            return 0.0
        exact = float(entry.metadata["event_prob"])
        tol = 1e-9
        if exact == 0:
            return 1.0 if val == 0 else 0.0
        if abs(val - exact) <= tol * max(1.0, abs(exact)):
            return 1.0
        return 0.0
