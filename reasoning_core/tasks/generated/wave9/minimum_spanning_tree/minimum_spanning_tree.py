"""Deterministic minimum-spanning-tree construction on weighted undirected graphs."""

import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


def _kruskal(n, edges):
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    total = 0
    used = []
    for u, v, w in sorted(edges, key=lambda e: (e[2], e[0], e[1])):
        if union(u, v):
            total += w
            used.append((u, v, w))
    return total, used


@dataclass
class MSTConfig(Config):
    n_vertices: int = 6
    min_w: int = 1
    max_w: int = 12
    tie_chance: float = 0.3

    def apply_difficulty(self, level):
        self.n_vertices = sround(self.n_vertices + level * 2)
        self.max_w = sround(self.max_w + level * 4)
        self.tie_chance = 0.3 + 0.1 * level


class MinimumSpanningTree(Task):
    summary = "Execute deterministic minimum-spanning-tree construction on weighted undirected graphs with ties, returning selected edges or total weight."

    config_cls = MSTConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_vertices

        while True:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    w = random.randint(cfg.min_w, cfg.max_w)
                    if random.random() < cfg.tie_chance * 0.6:
                        w = random.randint(cfg.min_w, cfg.max_w)
                    edges.append((u, v, w))
            random.shuffle(edges)
            total_w, used = _kruskal(n, edges)
            if total_w >= 0:
                break

        ans = str(total_w)

        metadata = edict({
            "n_vertices": n,
            "edges": [[u, v, w] for u, v, w in edges],
            "mst_weight": total_w,
        })
        metadata.payload = {
            "n_vertices": n,
            "edges": [[u, v, w] for u, v, w in edges],
        }
        return Entry(metadata=metadata, answer=ans)

    def render_prompt(self, metadata):
        lines = [f"Consider a connected weighted undirected graph on {metadata.n_vertices} vertices numbered 0 through {metadata.n_vertices - 1}."]
        lines.append("Its edges (u, v, w) with weight w are:")
        for u, v, w in metadata.edges:
            lines.append(f"  {u} -- {v}  weight {w}")
        lines.append(
            "Using Kruskal's algorithm, select a minimum spanning tree by always adding "
            "the smallest-weight edge that does not create a cycle, breaking ties by "
            "lowest vertex index (then the other endpoint). "
            "What is the total weight of the minimum spanning tree?"
        )
        lines.append("The answer is the integer total weight.")
        return "\n".join(lines) + "\n"

    def score_answer(self, answer, entry):
        try:
            got = float(answer.strip())
        except Exception:
            return 0.0
        return 1.0 if abs(got - float(entry.answer)) < 1e-9 else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'minimum_spanning_tree (draw 1 of 1)',
 'hypothesis': 'HV-013',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/minimum_spanning_tree',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3901837384,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
