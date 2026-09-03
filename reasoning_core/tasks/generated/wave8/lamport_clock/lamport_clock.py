import heapq
import random
import re
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


@dataclass
class LamportClockConfig(Config):
    n_processes: int = 3
    columns: int = 4
    num_messages: int = 4
    max_attempts: int = 500

    def apply_difficulty(self, level):
        self.n_processes = min(5, sround(self.n_processes + 0.35 * level))
        self.columns = sround(self.columns + level)
        self.num_messages = sround(self.num_messages + level)


def _compute_timestamps(n_proc, cols, messages):
    nid = lambda p, c: p * cols + c
    n = n_proc * cols
    incoming = [[] for _ in range(n)]
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for p in range(n_proc):
        for c in range(cols - 1):
            u, v = nid(p, c), nid(p, c + 1)
            adj[u].append(v)
            incoming[v].append(u)
            indeg[v] += 1
    for sp, sc, rp, rc in messages:
        u, v = nid(sp, sc), nid(rp, rc)
        adj[u].append(v)
        incoming[v].append(u)
        indeg[v] += 1
    heap = [(c, p, nid(p, c)) for p in range(n_proc) for c in range(cols) if indeg[nid(p, c)] == 0]
    heapq.heapify(heap)
    ts = [0] * n
    seen = 0
    while heap:
        c, p, u = heapq.heappop(heap)
        ts[u] = 1 + max((ts[w] for w in incoming[u]), default=0)
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            c2, p2 = v // cols, v % cols
            if indeg[v] == 0:
                heapq.heappush(heap, (c2, p2, v))
    if seen != n:
        return None
    return [[ts[nid(p, c)] for c in range(cols)] for p in range(n_proc)]


def _parse_int(text):
    m = re.search(r"-?\d+", str(text))
    return int(m.group()) if m else None


class LamportClock(Task):
    summary = (
        "Given per-process local event orders and cross-process message edges linking "
        "send to receive, compute the Lamport clock (happens-before) timestamp of a queried event."
    )
    config_cls = LamportClockConfig
    task_version = 2

    def generate_entry(self):
        n_proc = max(2, int(self.config.n_processes))
        cols = max(2, int(self.config.columns))
        n_msg = max(1, int(self.config.num_messages))
        for _ in range(int(self.config.max_attempts)):
            messages = set()
            for _ in range(n_msg):
                sp = random.randrange(n_proc)
                rp = random.randrange(n_proc - 1)
                if rp >= sp:
                    rp += 1
                sc = random.randrange(cols)
                rc = random.randrange(cols)
                messages.add((sp, sc, rp, rc))
            messages = sorted(messages)
            ts = _compute_timestamps(n_proc, cols, messages)
            if ts is None:
                continue
            candidates = [
                (p, c)
                for p in range(n_proc)
                for c in range(cols)
                if ts[p][c] != c + 1
            ]
            if not candidates:
                continue
            qp, qc = random.choice(candidates)
            gold = int(ts[qp][qc])
            if gold < 1:
                continue
            processes = {
                f"P{p + 1}": [f"e{c + 1}" for c in range(cols)]
                for p in range(n_proc)
            }
            edges = [
                f"P{sp + 1}.e{sc + 1} -> P{rp + 1}.e{rc + 1}" for sp, sc, rp, rc in messages
            ]
            query = f"P{qp + 1}.e{qc + 1}"
            payload = {
                "Local event orders (within a process, e1 happens before e2, then e3, and so on)": processes,
                "Message edges (sender -> receiver)": edges,
                "Queried event": query,
            }
            metadata = edict(
                n_processes=int(n_proc),
                n_columns=int(cols),
                num_messages=int(len(messages)),
                timestamps=[[int(v) for v in row] for row in ts],
                all_events=[
                    f"P{p + 1}.e{c + 1}" for p in range(n_proc) for c in range(cols)
                ],
                query=query,
            )
            metadata.payload = payload
            return Entry(metadata=metadata, answer=str(gold))
        raise RuntimeError("LamportClock: could not build a valid boosted-timestamp instance")

    def render_prompt(self, m):
        lines = [
            "Several processes run concurrent programs; within a single process events happen in strict "
            "order, so the first event e1 happens before e2, which happens before e3, and so on. A message "
            "edge 'sender -> receiver' records that the sender event sends a message that is received by "
            "the receiver event on another process, so the send happens-before the receive.",
            "Compute the Lamport clock timestamp of the queried event under the happens-before relation: "
            "each event's timestamp is one more than the largest timestamp among all events that "
            "happen-before it (its predecessor events, namely the prior event in its own process plus every "
            "event that sends a message into it); an event with no predecessors has timestamp 1.",
            render_payload(m.payload),
            "Answer with the Lamport timestamp of the queried event, as a single integer.",
        ]
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        val = _parse_int(answer)
        return 1.0 if val is not None and val == int(entry.answer) else 0.0

    def balancing_key(self, problem):
        return int(problem.answer)


TASK_META = {'parent_source_id': None,
 'idea': 'lamport_clock (draw 1 of 2)',
 'hypothesis': 'W1-035',
 'changes': 'new task in reasoning_core/tasks/generated/wave8/lamport_clock',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2791137760,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
