import random
import re
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'approval_voting_winner (draw 1 of 2)',
 'hypothesis': 'W1-065',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/approval_voting_winner',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3147129343,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ApprovalVotingWinnerConfig(Config):
    n_voters: int = 5
    n_candidates: int = 4

    def apply_difficulty(self, level):
        self.n_voters = sround(5 + 3 * level)
        self.n_candidates = sround(4 + level)


def _compute_winner(candidates, ballots):
    counts = {c: 0 for c in candidates}
    for ballot in ballots:
        for c in ballot:
            counts[c] += 1
    max_count = max(counts.values())
    tied = sorted(c for c in candidates if counts[c] == max_count)
    winner = tied[0]
    return winner, max_count, counts


def _parse_answer(answer):
    if not isinstance(answer, str):
        return None, None
    m = re.match(r"^\s*([A-Z])\s+(\d+)\s*$", answer)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


class ApprovalVotingWinner(Task):
    summary = ("Given approval ballots, output the winner with deterministic tie-breaking; "
               "vary voter count, candidate pool and approval density, answer carries winner letter and count.")
    config_cls = ApprovalVotingWinnerConfig
    task_version = 2

    def generate_entry(self):
        candidates = [chr(ord('A') + i) for i in range(self.config.n_candidates)]
        ballots = []
        for _ in range(self.config.n_voters):
            ballot = [c for c in candidates if random.random() < 0.5]
            if not ballot:
                ballot = [random.choice(candidates)]
            ballots.append(sorted(ballot))

        winner, max_count, counts = _compute_winner(candidates, ballots)
        assert max_count >= 1, "approval winner must have a strictly positive count"

        payload = {
            "candidates": " ".join(candidates),
            "ballots": "\n".join(f"{i + 1}: " + " ".join(b) for i, b in enumerate(ballots)),
        }
        metadata = edict({"payload": payload, "counts": counts, "winner": winner,
                          "approvals": max_count, "n_voters": self.config.n_voters,
                          "n_candidates": self.config.n_candidates})
        answer = f"{winner} {max_count}"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (f"In an approval election, voters approve any subset of the candidates. "
                f"Each line below is one ballot, listing the candidates that voter approves of.\n\n"
                f"{render_payload(metadata.payload)}\n\n"
                f"Count the approvals each candidate receives. Who is the winner? "
                f"If two or more candidates tie for the most approvals, the one that comes "
                f"first alphabetically is the winner.\n"
                f"The answer is the winner's letter followed by its number of approvals, "
                f"formatted as LETTER N, e.g. \"B 3\".")

    def score_answer(self, answer, entry):
        ref_letter, ref_count = _parse_answer(entry.answer)
        letter, count = _parse_answer(answer)
        if letter is None or count is None:
            return 0.0
        if letter == ref_letter and count == ref_count:
            return 1.0
        return 0.0
