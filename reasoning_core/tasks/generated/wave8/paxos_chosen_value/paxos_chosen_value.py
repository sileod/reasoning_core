import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'paxos_chosen_value (draw 1 of 2)',
 'hypothesis': 'W1-041',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/paxos_chosen_value',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2325212251,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class PaxosChosenValueConfig(Config):
    n_acceptors: int = 3
    n_proposals: int = 3
    max_value: int = 9

    def apply_difficulty(self, level):
        self.n_acceptors = max(3, sround(3 + level))
        self.n_proposals = sround(3 + 2 * level)
        self.max_value = 9 + 2 * level


def _quorum(n):
    return n // 2 + 1


class PaxosChosenValue(Task):
    summary = ("Given acceptor votes for numbered proposals, output the value already "
               "chosen by a quorum of acceptors, or None when no proposal has a quorum; "
               "varied acceptor counts, proposal counts, ballot numbering with ties broken "
               "by highest proposal number.")
    config_cls = PaxosChosenValueConfig

    def generate_entry(self):
        n = self.config.n_acceptors
        q = _quorum(n)
        n_prop = self.config.n_proposals
        max_v = max(int(self.config.max_value), 100)

        chosen = random.random() < 0.78
        assignment = [[] for _ in range(n_prop + 1)]
        chosen_p = None
        chosen_val = None

        if chosen:
            chosen_p = random.randint(1, n_prop)
            counts = [0] * (n_prop + 1)
            k = random.randint(q, max(q, n - 2))
            counts[chosen_p] = k
            remaining = n - k
            pool = [p for p in range(1, n_prop + 1) if p != chosen_p]
            random.shuffle(pool)
            for j, p in enumerate(pool):
                if remaining <= 0:
                    break
                if j == 0:
                    m = random.randint(1, min(q - 1, remaining))
                else:
                    m = random.randint(0, min(q - 1, remaining))
                counts[p] = m
                remaining -= m
            if remaining > 0 and pool:
                counts[pool[0]] += remaining
            else:
                counts[chosen_p] += remaining
            a = 0
            for p in range(1, n_prop + 1):
                for _ in range(counts[p]):
                    assignment[p].append(a)
                    a += 1
            chosen_val = random.randint(0, max_v)
            answer = str(chosen_val)
        else:
            counts = [0] * (n_prop + 1)
            a = 0
            for p in range(1, n_prop + 1):
                m = random.randint(0, q - 1)
                m = min(m, n - a)
                counts[p] = m
                a += m
                if a >= n:
                    break
            a = 0
            for p in range(1, n_prop + 1):
                for _ in range(counts[p]):
                    assignment[p].append(a)
                    a += 1
            answer = "None"

        rows = []
        for p in range(1, n_prop + 1):
            for acc in assignment[p]:
                v = chosen_val if p == chosen_p else random.randint(0, max_v)
                rows.append((acc, p, v))

        if chosen_val is not None:
            distractor = None
            for i, (acc, p, v) in enumerate(rows):
                if p != chosen_p:
                    distractor = i
                    break
            if distractor is not None:
                rows[distractor] = (rows[distractor][0], rows[distractor][1],
                                    random.randint(chosen_val + 1, chosen_val + max_v))
            low = None
            for i, (acc, p, v) in enumerate(rows):
                if p != chosen_p and i != distractor:
                    low = i
                    break
            if low is not None and chosen_val > 0:
                rows[low] = (rows[low][0], rows[low][1], random.randint(0, chosen_val - 1))
            random.shuffle(rows)
            for i in range(len(rows)):
                if rows[i][2] != chosen_val:
                    rows[-1], rows[i] = rows[i], rows[-1]
                    break
        else:
            random.shuffle(rows)
        lines = [f"Acceptor {a} voted for proposal {p} with value {v}." for (a, p, v) in rows]

        metadata = edict({})
        metadata.proposal_lines = lines
        metadata.quorum = q
        metadata.payload = {"votes": lines, "quorum_string": f"a quorum is {q} acceptors"}
        metadata.chosen = chosen_val
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        body = "\n".join(metadata.proposal_lines)
        return (f"In a Paxos round, {metadata.quorum} acceptors form a quorum. A value is "
                f"considered chosen if some proposal received accepted votes from at least a "
                f"quorum of acceptors. If several proposals have a quorum, the one with the "
                f"highest proposal number decides. Votes:\n{body}\n\n"
                f"What value is already chosen? Answer with the chosen value (an integer), "
                f"or the word None if no proposal has a quorum.")

    def score_answer(self, answer, entry):
        gold = entry.answer
        ans = str(answer).strip()
        if gold == "None":
            return 1.0 if ans == "None" else 0.0
        try:
            return 1.0 if int(ans) == int(gold) else 0.0
        except Exception:
            return 0.0
