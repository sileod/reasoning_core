from dataclasses import dataclass

import random as _random

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'Add multi-round voting rules over explicit ballots.',
 'hypothesis': 'S46',
 'changes': 'Ask for the elimination order under instant-runoff, or the full '
            'Borda score of each candidate.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 736651786,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _candidates(n):
    return [chr(ord('A') + i) for i in range(n)]


def _tally_firsts_for(ballots, remaining):
    from collections import Counter
    c = Counter()
    for ballot, mult in ballots:
        for cand in ballot:
            if cand in remaining:
                c[cand] += mult
                break
    return c


def _runoff_order(names, ballots):
    remaining = set(names)
    order = []
    while len(remaining) > 1:
        firsts = _tally_firsts_for(ballots, remaining)
        min_count = min(firsts.get(c, 0) for c in remaining)
        tied = [c for c in remaining if firsts.get(c, 0) == min_count]
        to_elim = min(tied)
        order.append(to_elim)
        remaining.discard(to_elim)
    order.append(sorted(remaining)[0])
    return order


def _runoff_winner(names, ballots):
    return _runoff_order(names, ballots)[-1]


def _borda(ballots, names):
    from collections import Counter
    n = len(names)
    scores = Counter()
    for ballot, mult in ballots:
        for idx, cand in enumerate(ballot):
            scores[cand] += (n - 1 - idx) * mult
    return dict(scores)


def _score_order(cleaned, entry):
    expected = [c.strip() for c in entry.answer.split(",")]
    got = [c.strip() for c in cleaned.replace("\n", ",").split(",")]
    return 1.0 if got == expected else 0.0


def _score_borda(cleaned, entry):
    names = set(entry.metadata.names)
    target = {}
    for pair in entry.answer.split(","):
        name, _, val = pair.partition("=")
        target[name] = int(val)
    parsed = {}
    ok = True
    for part in cleaned.replace("\n", ",").split(","):
        part = part.strip()
        name, eq, val = part.partition("=")
        name = name.strip()
        val = val.strip()
        if not eq or name not in names or name in parsed:
            ok = False
            break
        try:
            parsed[name] = int(val)
        except ValueError:
            ok = False
            break
    if not ok or set(parsed) != names:
        return 0.0
    correct = sum(1 for n in target if parsed.get(n) == target[n])
    return correct / len(target)


def score_answer(answer, entry):
    if answer is None:
        return 0.0
    cleaned = answer.strip()
    if not cleaned:
        return 0.0
    if entry.metadata.question == "borda":
        return _score_borda(cleaned, entry)
    return _score_order(cleaned, entry)


@dataclass
class VotingRulesConfig(Config):
    n_candidates: int = 4
    n_ballot_types: int = 5
    max_mult: int = 6
    question: str = "auto"

    def apply_difficulty(self, level):
        if level >= 5:
            self.n_candidates = 7
            self.n_ballot_types = 9
        elif level >= 3:
            self.n_candidates = 5
            self.n_ballot_types = 7
        else:
            self.n_candidates = 4
            self.n_ballot_types = 5
        self.max_mult = 6 + level


class VotingRules(Task):
    config_cls = VotingRulesConfig

    def generate_entry(self):
        cfg = self.config
        names = _candidates(cfg.n_candidates)
        n_types = min(cfg.n_ballot_types, cfg.n_candidates + 3)
        question = cfg.question
        if question not in ("runoff", "borda"):
            question = _random.choice(("runoff", "borda"))

        ballots = []
        for _ in range(n_types):
            ranking = list(names)
            _random.shuffle(ranking)
            mult = _random.randint(1, cfg.max_mult)
            ballots.append((tuple(ranking), int(mult)))

        if question == "borda":
            scores = _borda(ballots, names)
            ordered = sorted((scores[n], n) for n in names)
            answer = ",".join("%s=%d" % (n, s) for s, n in ordered)
        else:
            order = _runoff_order(names, ballots)
            answer = ",".join(order)

        ballot_lines = []
        for ranking, mult in ballots:
            ballot_lines.append("%s voters: %s > %s > %s"
                                % (mult, ranking[0], ranking[1], " > ".join(ranking[2:])))

        metadata = edict({
            "n_candidates": int(cfg.n_candidates),
            "names": list(names),
            "ballots": ballots,
            "question": question,
        })
        metadata.payload = {
            "candidates": list(names),
            "ballot_lines": ballot_lines,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        q = metadata.question
        body = render_payload(metadata.payload)
        if q == "borda":
            return (f"{body}\n\n"
                    f"Each ballot lists candidates from most preferred to least preferred; "
                    f"the quoted number is how many voters submitted that ranking. Compute "
                    f"the Borda score of every candidate: {metadata.n_candidates-1} points per "
                    f"first-place vote, {metadata.n_candidates-2} per second-place vote, down "
                    f"to 0 for last place. Give every candidate in alphabetical order as "
                    f"exact strings in the format:\nBorda: A=s, B=s, ...\n"
                    f"(one per candidate, in that order).")
        return (f"{body}\n\n"
                f"Each ballot lists candidates from most preferred to least preferred; the "
                f"quoted number is how many voters submitted that ranking. In instant-runoff "
                f"voting, each round counts only each voter's highest-ranked remaining "
                f"candidate. Eliminate the candidate with the fewest first-place votes; ties "
                f"break by eliminating the alphabetically first. Repeat until one remains. "
                f"Give the elimination order, first eliminated to winner, as a "
                f"comma-separated list of candidates:\nOrder: X, Y, Z, ...\n")

    def score_answer(self, answer, entry):
        return score_answer(answer, entry)
