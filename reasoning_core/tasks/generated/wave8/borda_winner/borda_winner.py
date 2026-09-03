import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'borda_winner (draw 1 of 2)',
 'hypothesis': 'W1-061',
 'changes': 'new task in reasoning_core/tasks/generated/wave8/borda_winner',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2916049189,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class BordaWinnerConfig(Config):
    n_candidates: int = 5
    n_voters: int = 4

    def apply_difficulty(self, level):
        self.n_candidates = sround(self.n_candidates + level)
        self.n_voters = sround(self.n_voters + level)


def _borda_scores(labels, ballots):
    n = len(labels)
    scores = {c: 0 for c in labels}
    for ballot in ballots:
        for pos, c in enumerate(ballot):
            scores[c] += (n - 1) - pos
    return scores


def _solve(labels, ballots):
    scores = _borda_scores(labels, ballots)
    best = max(scores.values())
    winners = [c for c in labels if scores[c] == best]
    return min(winners)


def _rows(ballots):
    return "\n".join(f"Voter {i + 1}: " + " > ".join(b) for i, b in enumerate(ballots))


class BordaWinner(Task):
    summary = "Given ranked ballots, output the Borda winner with deterministic tie-breaking (lexicographically smallest among tied; ties broken by first preference)."
    config_cls = BordaWinnerConfig

    def generate_entry(self):
        cfg = self.config
        n = max(3, cfg.n_candidates)
        labels = [chr(65 + i) for i in range(n)]
        ballots = [random.sample(labels, len(labels)) for _ in range(max(2, cfg.n_voters))]
        winner = _solve(labels, ballots)
        return Entry(metadata=edict(labels=labels, ballots=ballots), answer=winner)

    def render_prompt(self, m):
        return (
            f"Candidates: {', '.join(m.labels)}\n"
            f"{_rows(m.ballots)}\n"
            "Borda count: a candidate ranked position p from the top (first = 0) gets "
            f"({len(m.labels)} - 1 - p) points per voter, where a larger number means a better rank. "
            "Sum each candidate's points across all voters, then pick the candidate with the most points. "
            "If several are tied for the most, the winner is the lexicographically smallest candidate name "
            "(the first one alphabetically).\n"
            "The answer is the winning candidate's single letter."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return float(answer.strip().upper() == entry.answer)
