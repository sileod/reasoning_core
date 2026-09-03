import random

from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload
from reasoning_core.utils import score_scalar


@dataclass
class CondorcetWinnerConfig(Config):
    n_candidates: int = 4
    n_ballots: int = 8
    max_ballots_per_group: int = 6

    def apply_difficulty(self, level):
        self.n_candidates = 3 + level
        self.n_ballots = 4 + 2 * level
        self.max_ballots_per_group = 3 + level


def _parse_answer(answer):
    try:
        a = answer.strip().lower()
    except AttributeError:
        return None
    if a in ("none", "no condorcet winner", "no winner"):
        return ("none",)
    try:
        v = int(a)
    except (ValueError, TypeError):
        return None
    return v


class CondorcetWinner(Task):
    summary = (
        "Given ranked preference ballots over candidates, determine whether a "
        "Condorcet winner exists (the candidate who beats every other in a "
        "majority of pairwise contests) and output that candidate or None."
    )
    config_cls = CondorcetWinnerConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        rng = random
        n = cfg.n_candidates
        candidates = list(range(n))
        # Use an odd ballot count so strict pairwise majorities never tie.
        n_ballots = cfg.n_ballots if cfg.n_ballots % 2 == 1 else cfg.n_ballots + 1
        half = n_ballots / 2.0

        def pair_counts_of(orders):
            pc = [[0] * n for _ in range(n)]
            for order in orders:
                pos = {c: i for i, c in enumerate(order)}
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        if pos[a] < pos[b]:
                            pc[a][b] += 1
            return pc

        def find_winners(pc):
            ws = []
            for a in range(n):
                ok = True
                for b in range(n):
                    if b == a:
                        continue
                    if not (pc[a][b] > half):
                        ok = False
                        break
                if ok:
                    ws.append(a)
            return ws

        while True:
            # Roughly half the instances have a Condorcet winner (constructed),
            # half do not (drawn randomly). This keeps the answer space wide:
            # numeric candidate indices plus None.
            if rng.random() < 0.7:
                # Construct an instance with a chosen Condorcet winner w.
                w = rng.randrange(n)
                orders = []
                used = set()
                lead = n_ballots // 2 + 1  # ballots placing w first (strict majority)
                for _ in range(lead):
                    rest = [c for c in candidates if c != w]
                    rng.shuffle(rest)
                    order = [w] + rest
                    t = tuple(order)
                    if t in used:
                        continue
                    used.add(t)
                    orders.append(order)
                while len(orders) < n_ballots:
                    rest = [c for c in candidates if c != w]
                    rng.shuffle(rest)
                    rng.shuffle(rest)
                    order = rest[:]
                    order.insert(rng.randrange(n), w)
                    t = tuple(order)
                    if t in used:
                        continue
                    used.add(t)
                    orders.append(order)
                pc = pair_counts_of(orders)
                ws = find_winners(pc)
                if ws == [w]:
                    outcome = w
                    break
                # If construction failed (should not happen), redraw.
            else:
                # Draw fully random rankings, expected to usually have no
                # Condorcet winner; accept only when that holds.
                orders = []
                used = set()
                while len(orders) < n_ballots:
                    perm = candidates[:]
                    rng.shuffle(perm)
                    t = tuple(perm)
                    if t in used:
                        continue
                    used.add(t)
                    orders.append(perm)
                pc = pair_counts_of(orders)
                if find_winners(pc) == []:
                    outcome = None
                    break

        answer = str(outcome) if outcome is not None else "None"

        # Verify from scratch that the claimed answer is right.
        assert find_winners(pc) == ([] if outcome is None else [outcome])

        metadata = edict({
            "ballots": orders,
            "n_candidates": n,
        })
        metadata.payload = {
            "ballots": orders,
            "n_candidates": n,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        cands = ", ".join(str(c) for c in range(metadata.n_candidates))
        ballots_txt = "\n".join(
            f"Ballot {i+1}: {', '.join(str(c) for c in b)}"
            for i, b in enumerate(metadata.ballots)
        )
        return (
            f"As a voting official, consider an election over candidates "
            f"{cands}. Each ballot is a strict ranking of every candidate from "
            f"most preferred to least preferred. A candidate is the Condorcet "
            f"winner if, in a pairwise contest against every other candidate, "
            f"they are ranked above that candidate on a strict majority of the "
            f"ballots.\n\n"
            f"{ballots_txt}\n\n"
            f"Output the Condorcet winner as a single number (0-based candidate "
            f"index), or output None if no Condorcet winner exists."
        )

    def score_answer(self, answer, entry):
        parsed = _parse_answer(answer)
        if parsed is None:
            return 0.0
        gold = _parse_answer(entry.answer)
        return 1.0 if parsed == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'condorcet_winner (draw 1 of 2)',
 'hypothesis': 'W1-063',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/condorcet_winner',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 259485130,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
