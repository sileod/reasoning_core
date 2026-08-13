import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


@dataclass
class DynamicProgrammingConfig(Config):
    n_states: int = 3
    n_symbols: int = 3
    length: int = 5
    magnitude: int = 3

    def apply_difficulty(self, level):
        self.n_states = sround(self.n_states + 0.35 * level)
        self.n_symbols = sround(self.n_symbols + 0.2 * level)
        self.length = sround(self.length + 1.2 * level)
        self.magnitude = sround(self.magnitude + 0.4 * level)


def _solve(obs, start, trans, emit):
    dp = [(start[s] + emit[s][obs[0]], (s,)) for s in range(len(start))]
    for o in obs[1:]:
        nxt = []
        for s in range(len(start)):
            candidates = [(dp[p][0] + trans[p][s] + emit[s][o], dp[p][1] + (s,)) for p in range(len(start))]
            score = max(x[0] for x in candidates)
            nxt.append((score, min(path for value, path in candidates if value == score)))
        dp = nxt
    score = max(x[0] for x in dp)
    return min(path for value, path in dp if value == score)


def _rows(labels, matrix):
    return "\n".join(f"{s}: " + " ".join(map(str, row)) for s, row in zip(labels, matrix))


class DynamicProgrammingMixin(GeneratedMixin):
    summary = "Evaluate a max-sum dynamic program and reconstruct its optimal state sequence."
    config_cls = DynamicProgrammingConfig

    def generate_entry(self):
        cfg = self.config
        labels = [chr(65 + i) for i in range(cfg.n_states)]
        r = range(cfg.n_states)
        m = cfg.magnitude
        start = [random.randint(-m, m) for _ in r]
        trans = [[random.randint(-m, m) for _ in r] for _ in r]
        emit = [[random.randint(-m, m) for _ in range(cfg.n_symbols)] for _ in r]
        obs = [random.randrange(cfg.n_symbols) for _ in range(cfg.length)]
        path = _solve(obs, start, trans, emit)
        return Entry(metadata=edict(labels=labels, obs=obs, start=start, trans=trans, emit=emit), answer=" ".join(labels[i] for i in path))

    def render_prompt(self, m):
        labels = " ".join(m.labels)
        return (
            f"States: {labels}\nObservations: {' '.join(map(str, m.obs))}\n"
            f"Start: {' '.join(f'{s}={v}' for s, v in zip(m.labels, m.start))}\n"
            f"Transitions (rows=from, columns={labels}):\n{_rows(m.labels, m.trans)}\n"
            f"Emissions (rows=state, columns=0..{len(m.emit[0])-1}):\n{_rows(m.labels, m.emit)}\n"
            "Score a state sequence by start + emissions + transitions. Find the maximum-score sequence; ties are lexicographic. "
            "The answer is the space-separated state labels."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).upper().replace(",", " ").split())
        return float(norm(answer) == norm(entry.answer))

    def balancing_key(self, problem):
        return problem.answer.split()[0]
