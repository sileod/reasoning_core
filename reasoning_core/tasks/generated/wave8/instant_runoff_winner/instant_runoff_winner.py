import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'instant_runoff_winner (draw 1 of 2)',
 'hypothesis': 'W1-062',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/instant_runoff_winner',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1501836147,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _irv_winner(ballots, n_candidates):
    """Reference IRV implementation. Returns winner's 0-based id.

    Rounds: count first preferences among remaining; strict majority of the active
    votes wins. Else eliminate the remaining candidate with the fewest first
    preferences, tie broken by lowest candidate id. Transfer eliminated ballots to
    their next remaining preference. Ballots with no remaining preference are
    exhausted (dropped). When two remain and neither holds the majority of active
    votes, the one with more active votes wins.
    """
    remaining = list(range(n_candidates))
    for _ in range(n_candidates):
        counts = [0] * n_candidates
        for ballot in ballots:
            for b in ballot:
                if b in remaining:
                    counts[b] += 1
                    break
        active = sum(counts[c] for c in remaining)
        for c in remaining:
            if counts[c] * 2 > active:
                return c
        if len(remaining) == 2:
            a, b = remaining
            return a if counts[a] >= counts[b] else b
        min_votes = min(counts[c] for c in remaining)
        lowest = sorted(c for c in remaining if counts[c] == min_votes)
        remaining.remove(lowest[0])


def _irv_winner2(ballots, n_candidates):
    """Independent re-derivation using a set-based active loop."""
    active = set(range(n_candidates))
    while True:
        counts = {c: 0 for c in active}
        for ballot in ballots:
            for b in ballot:
                if b in active:
                    counts[b] += 1
                    break
        total = sum(counts.values())
        for c in list(counts):
            if counts[c] * 2 > total:
                return c
        if len(active) == 2:
            a, b = sorted(active)
            return a if counts[a] >= counts[b] else b
        m = min(counts.values())
        elim = min(c for c in active if counts[c] == m)
        active.remove(elim)


@dataclass
class InstantRunoffConfig(Config):
    n_candidates: int = 3
    n_ballots: int = 5
    n_ranks: int = 3

    def apply_difficulty(self, level):
        self.n_candidates = sround(3 + level)
        self.n_ballots = sround(5 + 2 * level)
        self.n_ranks = sround(3 + level)


class InstantRunoffWinner(Task):
    summary = "Given complete ranked-ballot profiles, execute instant-runoff elimination (majority, elimination of the plurality-minimum with tie-break by lowest candidate id, vote transfer, exhausted ballots) and output the winning candidate id; answers are candidate ids across varied candidate, ballot and rank counts."
    config_cls = InstantRunoffConfig
    task_version = 2

    def generate_entry(self):
        n_c = self.config.n_candidates
        n_b = self.config.n_ballots
        n_r = max(2, min(self.config.n_ranks, n_c))
        candidates = list(range(n_c))
        ballots = []
        for _ in range(1000):
            ballots = []
            for _ in range(n_b):
                perm = candidates[:]
                random.shuffle(perm)
                ballots.append(perm[:n_r])
            w1 = _irv_winner(ballots, n_c)
            w2 = _irv_winner2(ballots, n_c)
            if w1 != w2:
                continue
            counts = [0] * n_c
            for b in ballots:
                counts[b[0]] += 1
            active_first = [c for c in candidates if counts[c] > 0]
            if len(active_first) < 2:
                continue
            if any(counts[c] * 2 > n_b for c in candidates):
                continue
            if w1 == candidates[-1]:
                continue
            winner = w1
            break
        else:
            raise RuntimeError("could not build a non-degenerate IRV instance")
        names = [CHR[i] for i in candidates]
        rows = ["".join(names[b] for b in ballot) for ballot in ballots]
        metadata = edict({
            "n_candidates": int(n_c),
            "candidate_names": names,
            "ballots": rows,
            "winner": int(winner),
        })
        metadata.payload = {
            "candidates": ", ".join(names),
            "ballots": "\n".join(rows),
        }
        return Entry(metadata=metadata, answer=names[winner])

    def render_prompt(self, metadata):
        payload = metadata.payload
        return (
            f"{render_payload(payload)}\n\n"
            "Candidates are the letters listed (ordered A, B, C, ...). Each ballot is an "
            "ordered ranking, most preferred first. Apply instant-runoff (IRV). In each "
            "round count first-preference votes among the still-running candidates; if any "
            "of them holds a strict majority of the active votes, they win. Otherwise "
            "eliminate the running candidate with the fewest first-preference votes, "
            "breaking ties by eliminating the one whose letter is alphabetically earliest, "
            "then transfer each eliminated candidate's ballots to the next still-running "
            "preference on that ballot. A ballot with no still-running preference is "
            "exhausted. If two candidates remain with no strict majority, the one with more "
            "active votes wins; if tied, the alphabetically earlier wins. "
            "Output the winner's letter alone."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        ans = str(answer).strip().upper()
        gold = str(entry.answer)
        return 1.0 if ans == gold else 0.0


CHR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
