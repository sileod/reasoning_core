"""Given a wait-for graph, output the transactions participating in deadlock cycles."""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'wait_for_deadlock (draw 1 of 2)',
 'hypothesis': 'W1-039',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/wait_for_deadlock',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1977587174,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _find_deadlocked(txns, edges):
    """Return the set of transaction labels that participate in at least one directed cycle.

    A transaction is deadlocked if it can reach itself through at least one edge,
    i.e. it lies on a directed cycle.
    """
    n = txns
    adj = {i: [] for i in range(n)}
    for (a, b) in edges:
        adj[a].append(b)

    deadlocked = set()

    def reaches(start, target):
        # Depth-first search: is there a directed path start -> target of length >= 1?
        stack = [start]
        seen = set()
        while stack:
            node = stack.pop()
            for nxt in adj[node]:
                if nxt == target:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    for v in range(n):
        # v is on a cycle iff some outgoing edge from v leads back to v.
        if any(reaches(w, v) for w in adj[v]):
            deadlocked.add(v)
    return deadlocked


@dataclass
class WaitForDeadlockConfig(Config):
    n_txns: int = 6
    n_edges: int = 6
    p_extra: float = 0.4

    def apply_difficulty(self, level):
        self.n_txns = sround(self.n_txns + level)
        self.n_edges = sround(self.n_edges + level)
        self.p_extra = min(0.9, 0.4 + 0.08 * level)


class WaitForDeadlock(Task):
    summary = ("Given a wait-for graph whose nodes are transactions and whose edges are waits, "
               "output the transaction label set that participates in at least one directed deadlock "
               "cycle, in numeric order.")
    config_cls = WaitForDeadlockConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_txns)
        half_cycle = int(max(2, n // 2))
        while True:
            edges = set()
            # A "chain" that snakes both ways so a cycle can close easily.
            perm = list(range(n))
            random.shuffle(perm)
            for i in range(len(perm) - 1):
                edges.add((perm[i], perm[i + 1]))
            # Randomly close a few chords that may create cycles.
            n_edges = int(cfg.n_edges)
            while len(edges) < n_edges:
                a = random.randrange(n)
                b = random.randrange(n)
                if a != b:
                    edges.add((a, b))
            # With some probability force at least one cycle by adding a back edge.
            if random.random() < cfg.p_extra and len(edges) < n * (n - 1):
                guard = 0
                while guard < 50:
                    guard += 1
                    a = random.randrange(half_cycle)
                    b = random.randrange(half_cycle)
                    if a != b:
                        edges.add((b, a))
                        break
            edges = sorted(edges)
            dead = sorted(_find_deadlocked(n, edges))
            # Enforce domain: answer is a set of transaction labels within [0, n-1].
            if not all(0 <= d <= n - 1 for d in dead):
                continue
            # Keep a healthy mix: reject all-deadlocked and all-clear extremes rarely.
            if not dead and all(b <= a for (a, b) in edges):
                continue
            break

        labels = [f"T{d}" for d in range(n)]
        edge_lines = [f"{labels[a]} waits for {labels[b]}" for (a, b) in edges]
        answer = ",".join(f"T{d}" for d in dead)
        if not dead:
            answer = "none"
        metadata = edict({
            "n_txns": n,
            "edges": [(int(a), int(b)) for (a, b) in edges],
            "dead": [int(d) for d in dead],
        })
        metadata.payload = {
            "transactions": labels,
            "wait_edges": edge_lines,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        body = render_payload(metadata.payload)
        return (body +
                "\n\nSome transactions may be stuck in a deadlock cycle: a cycle of "
                "transactions where each waits for a resource the next holds. A transaction "
                "participates in a deadlock if it lies on at least one directed cycle of "
                "wait_edges.\n\n"
                "List the labels of all deadlocked transactions, comma-separated in numeric "
                "order (for example \"T1,T3,T5\"), or exactly \"none\" if no transaction is "
                "deadlocked.")

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        if a == gold:
            return 1.0
        return 0.0
