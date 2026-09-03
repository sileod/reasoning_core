import random

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'dhondt_apportionment (draw 1 of 2)',
 'hypothesis': 'W1-064',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/dhondt_apportionment',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3332846379,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _dhondt(votes, seats):
    n = len(votes)
    assigned = [0] * n
    for _ in range(seats):
        best = -1
        best_key = None
        for i in range(n):
            q = votes[i] / (assigned[i] + 1)
            key = (q, i)
            if best_key is None or key > best_key:
                best_key = key
                best = i
        assigned[best] += 1
    return assigned


def _parse_answer(answer):
    s = answer.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


class DhondtApportionmentV1Config(Config):
    n_parties: int = 2
    seats: int = 8

    def apply_difficulty(self, level):
        self.n_parties = 2 + 2 * level
        self.seats = 8 + 4 * level


class DhondtApportionment(Task):
    summary = ("Given party vote totals and a seat count, output the final "
               "D'Hondt seat vector across parties over varied party counts, "
               "seat counts and vote magnitudes.")
    config_cls = DhondtApportionmentV1Config
    task_version = 2

    def generate_entry(self):
        n = self.config.n_parties
        seats = self.config.seats
        while True:
            votes = [random.randint(1, 2000) for _ in range(n)]
            if len(set(votes)) < n:
                continue
            assigned = _dhondt(votes, seats)
            if sum(assigned) != seats:
                continue
            break
        assert all(v >= 1 for v in votes), votes
        assert all(a >= 0 for a in assigned), assigned
        assert sum(assigned) == seats, (assigned, seats)
        votes_out = ",".join(str(v) for v in votes)
        answer = ",".join(str(a) for a in assigned)
        metadata = edict({
            "votes": votes,
            "seats": seats,
            "n_parties": n,
        })
        metadata.payload = {
            "votes": votes_out,
            "seats": seats,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        votes = metadata.payload["votes"]
        seats = metadata.payload["seats"]
        n = metadata.n_parties
        return (
            f"An election assigns {seats} seats among {n} parties using the "
            f"D'Hondt method. The party vote totals are [{votes}]. Using the "
            f"D'Hondt highest-average rule (a party receives a seat whenever "
            f"its votes divided by one plus its current seats is the largest "
            f"such quotient, with ties broken toward the earlier-listed "
            f"party), what is the final seat vector for the parties in the "
            f"same order as the vote list? "
            f"Give the answer as a comma-separated list of non-negative "
            f"integers inside square brackets, one seat count per party, "
            f"for example [1,5]."
        )

    def score_answer(self, answer, entry):
        if isinstance(answer, str):
            try:
                got = _parse_answer(answer)
            except Exception:
                return 0.0
        else:
            return 0.0
        want = _parse_answer(entry.answer)
        if len(got) != len(want):
            return 0.0
        if got == want:
            return 1.0
        return 0.0
