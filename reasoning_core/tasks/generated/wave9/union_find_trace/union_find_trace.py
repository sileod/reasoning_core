import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'union_find_trace (draw 1 of 1)',
 'hypothesis': 'HV-019',
 'changes': 'new task in reasoning_core/tasks/generated/wave9/union_find_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1602037825,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def uf_connectivity(n_nodes, edges):
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

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

    for (a, b) in edges:
        union(a, b)
    return find


@dataclass
class UFConfig(Config):
    n_nodes: int = 8
    n_edges: int = 8
    n_queries: int = 3

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 2 * level)
        self.n_edges = sround(self.n_edges + 3 * level)
        self.n_queries = sround(self.n_queries + level)


class UnionFindTrace(Task):
    summary = "Execute union-find operations under stated union-by-rank and path-compression rules, returning connectivity or canonical parent and rank state."

    config_cls = UFConfig

    def generate_entry(self):
        while True:
            n = self.config.n_nodes
            m = self.config.n_edges
            k = self.config.n_queries
            edges = [(random.randrange(n), random.randrange(n)) for _ in range(m)]
            queries = [(random.randrange(n), random.randrange(n)) for _ in range(k)]

            # Do a plain union-find (with proper ranks) to get final components.
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
                    return
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[ra] > rank[rb]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1

            for (a, b) in edges:
                union(a, b)

            # Deterministically map each node to the min node in its component (canonical rep).
            comp_min = {}
            for x in range(n):
                r = find(x)
                comp_min.setdefault(r, x)
                comp_min[r] = min(comp_min[r], x)
            canon = {}
            for x in range(n):
                canon[x] = comp_min[find(x)]

            # Answer: for each query, whether the two are connected, then the canonical
            # root of the first query node. Pack into a single integer code.
            answers = []
            for (a, b) in queries:
                conn = 1 if find(a) == find(b) else 0
                answers.append(conn)
                answers.append(canon[a])

            # Verify with a disjoint independent check (component rep via DFS/union).
            adj = [[] for _ in range(n)]
            for (a, b) in edges:
                adj[a].append(b)
                adj[b].append(a)
            vis = [False] * n
            comp = [-1] * n
            cid = 0
            for s in range(n):
                if not vis[s]:
                    stack = [s]
                    vis[s] = True
                    while stack:
                        u = stack.pop()
                        comp[u] = cid
                        for v in adj[u]:
                            if not vis[v]:
                                vis[v] = True
                                stack.append(v)
                    cid += 1
            for (a, b) in queries:
                assert (comp[a] == comp[b]) == (find(a) == find(b))
            # canonical min consistent
            comp_nodes = {}
            for x in range(n):
                comp_nodes.setdefault(comp[x], []).append(x)
            for c in comp_nodes.values():
                mn = min(c)
                for x in c:
                    assert canon[x] == mn

            answer = " ".join(str(a) for a in answers)

            if len(set(answers)) >= 3:
                break

        metadata = edict({
            "n_nodes": n,
            "edges": [[int(a), int(b)] for (a, b) in edges],
            "queries": [[int(a), int(b)] for (a, b) in queries],
            "canonical": [canon[i] for i in range(n)],
            "connectivity": [int(find(a) == find(b)) for (a, b) in queries],
        })
        metadata.payload = {
            "nodes": [i for i in range(n)],
            "edges": [[int(a), int(b)] for (a, b) in edges],
            "queries": [[int(a), int(b)] for (a, b) in queries],
            "rules": "union-by-rank with path compression",
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        lines = []
        lines.append("A graph has %d nodes numbered 0 through %d." % (p["nodes"][-1] + 1, p["nodes"][-1]))
        lines.append("Its edges, added in this order, are: %s." % (render_payload({"edges": p["edges"]})))
        lines.append("We maintain a union-find data structure over the nodes using union-by-rank with path compression.")
        lines.append("Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.")
        lines.append("For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.")
        lines.append("Queries: %s." % (render_payload({"queries": p["queries"]})))
        lines.append("Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        ref = entry.answer.split(" ")
        user = answer.split()
        if len(user) != len(ref):
            return 0.0
        try:
            user_ints = [int(x) for x in user]
        except ValueError:
            return 0.0
        ref_ints = [int(x) for x in ref]
        if user_ints == ref_ints:
            return 1.0
        return 0.0
