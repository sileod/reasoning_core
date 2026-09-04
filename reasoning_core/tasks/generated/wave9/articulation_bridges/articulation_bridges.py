"""Compute DFS low-link information in undirected graphs and return articulation
vertices, bridge edges, or a queried connectivity consequence."""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'articulation_bridges (draw 1 of 1)',
 'hypothesis': 'HV-016',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/articulation_bridges',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1881068873,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ArticulationBridgesConfig(Config):
    n_verts: int = 9
    mode: int = 0  # 0=articulation, 1=bridges, 2=connectivity consequence
    edge_p: float = 0.34

    def apply_difficulty(self, level):
        self.n_verts = sround(self.n_verts + level * 2)
        self.edge_p = 0.34 + 0.01 * level


def _tarjan(adj, n):
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    ap = set()
    bridges = set()
    time = 0

    def dfs(u):
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        children = 0
        for v in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    ap.add(u)
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap.add(u)
                if low[v] > disc[u]:
                    bridges.add((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i)
    return ap, bridges


def _connected_components(adj, n, removed):
    seen = set()
    removed = set(removed)
    comps = 0
    for s in range(n):
        if s in removed or s in seen:
            continue
        comps += 1
        stack = [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in removed or v in seen:
                    continue
                seen.add(v)
                stack.append(v)
    return comps


def _cc_without(adj, n, cut):
    """Number of connected components after removing vertex `cut` (and incident edges)."""
    removed = [cut]
    return _connected_components(adj, n, removed)


class ArticulationBridges(Task):
    summary = ("Query articulation vertices, bridge edges, or connectivity consequences "
               "(number of components after removing one vertex) on undirected graphs "
               "using DFS low-link; answer is a sorted vertex/edge list or an integer.")
    config_cls = ArticulationBridgesConfig
    task_version = 2

    def generate_entry(self):
        n = int(self.config.n_verts)
        r = random.random()
        if r < 0.45:
            mode = 0
        elif r < 0.80:
            mode = 1
        else:
            mode = 2
        edge_p = self.config.edge_p
        tgt = -1

        while True:
            adj = [[] for _ in range(n)]
            edges = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < edge_p:
                        adj[u].append(v)
                        adj[v].append(u)
                        edges.add((u, v))
            if not edges:
                continue
            ap, bridges = _tarjan(adj, n)
            if mode == 0:
                if not ap:
                    continue
                answer = ",".join(str(v) for v in sorted(ap))
                label = "articulation"
            elif mode == 1:
                if not bridges:
                    continue
                bridge_list = [tuple(sorted(e)) for e in bridges]
                answer = ";".join(f"{u}-{v}" for u, v in sorted(bridge_list))
                label = "bridge"
            else:
                # connectivity consequence: components after removing one articulation vertex
                if not ap:
                    continue
                tgt = min(ap)
                comps = _cc_without(adj, n, tgt)
                answer = str(comps)
                label = "components"
            break

        metadata = edict({
            "n": n,
            "edges": sorted(edges),
            "mode": mode,
            "label": label,
            "target": tgt,
        })
        metadata.payload = {
            "vertices": list(range(n)),
            "edges": sorted(edges),
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        if metadata.mode == 2:
            question = ("After removing vertex {v} (and all its incident edges), the remaining graph "
                        "splits into some number of connected components. What is that number?").format(v=metadata.target)
            return render_payload(metadata.payload) + "\n\n" + question + "\nThe answer is a single non-negative integer."
        if metadata.mode == 0:
            return (render_payload(metadata.payload)
                    + "\n\nFind all articulation vertices (vertices whose removal increases the number "
                      "of connected components) using the DFS low-link algorithm. List them as a "
                      "comma-separated sequence in increasing order. "
                      "The answer is that comma-separated list, or the single word none if there are none.")
        return (render_payload(metadata.payload)
                + "\n\nFind all bridge edges (edges whose removal increases the number of connected "
                  "components) using the DFS low-link algorithm. List each as u-v with the smaller "
                  "endpoint first, and the edges separated by semicolons in increasing lexicographic "
                  "order. The answer is that semicolon-separated list, or the single word none if there are none.")

    def score_answer(self, answer, entry):
        want = entry.answer
        if entry.metadata.mode == 2:
            try:
                got = int(str(answer).strip())
            except (ValueError, TypeError):
                return 0.0
            return 1.0 if got == int(want) else 0.0
        if answer is None:
            return 0.0
        return 1.0 if str(answer).strip() == want else 0.0
