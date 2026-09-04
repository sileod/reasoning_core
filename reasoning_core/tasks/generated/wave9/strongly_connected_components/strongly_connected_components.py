import random
from dataclasses import dataclass

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'strongly_connected_components (draw 1 of 1)',
 'hypothesis': 'HV-011',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/strongly_connected_components',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2305351643,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def kosaraju_scc(n, edges):
    g = [[] for _ in range(n)]
    gr = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        gr[v].append(u)
    visited = [False] * n
    order = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            node, idx = stack[-1]
            if idx < len(g[node]):
                nxt = g[node][idx]
                stack[-1] = (node, idx + 1)
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, 0))
            else:
                order.append(node)
                stack.pop()
    seen = [False] * n
    comps = []
    for node in reversed(order):
        if seen[node]:
            continue
        comp = []
        stack = [node]
        seen[node] = True
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in gr[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        comp.sort()
        comps.append(comp)
    comps.sort(key=lambda c: c[0])
    return comps


@dataclass
class StronglyConnectedComponentsConfig(Config):
    n_nodes: int = 7
    extra_edges: int = 6
    comp_min: int = 2
    comp_max: int = 4
    mode: str = "component"

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + level * 2)
        self.extra_edges = sround(self.extra_edges + level * 2)
        self.comp_min = 2
        self.comp_max = 3 + level


class StronglyConnectedComponents(Task):
    summary = ("Partition directed graphs into strongly connected components; "
               "return the canonical (min-label) component containing a queried "
               "node, else the whole canonical partition of the graph into "
               "strongly connected components.")
    config_cls = StronglyConnectedComponentsConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nodes
        mode_map = ["component", "partition"]
        mode = mode_map[random.randrange(2)]

        while True:
            comp_sizes = []
            remaining = n
            while remaining > 0:
                if remaining <= cfg.comp_max:
                    comp_sizes.append(remaining)
                    remaining = 0
                    break
                size = random.randint(cfg.comp_min, cfg.comp_max)
                leftover = remaining - size
                if 0 < leftover < cfg.comp_min:
                    continue
                comp_sizes.append(size)
                remaining = leftover
            if len(comp_sizes) >= 2:
                break

        nodes = list(range(n))
        random.shuffle(nodes)
        comps_nodes = []
        idx = 0
        for s in comp_sizes:
            comps_nodes.append(sorted(nodes[idx:idx + s]))
            idx += s

        edges = []
        for comp in comps_nodes:
            if len(comp) >= 2:
                for i in range(len(comp)):
                    edges.append((comp[i], comp[(i + 1) % len(comp)]))
            else:
                edges.append((comp[0], comp[0]))
                continue
            for _ in range(random.randint(0, 1)):
                a = random.choice(comp)
                bs = [b for b in comp if b != a]
                if not bs:
                    continue
                edges.append((a, random.choice(bs)))

        max_inter_edges = len(comps_nodes) * len(comps_nodes)
        extra = cfg.extra_edges
        inter_placed = 0
        order = list(range(len(comps_nodes)))
        random.shuffle(order)
        tries = 0
        for _ in range(extra):
            if len(comps_nodes) < 2:
                break
            i = random.randrange(len(order) - 1)
            j = random.randrange(i + 1, len(order))
            comp_u = comps_nodes[order[i]]
            comp_v = comps_nodes[order[j]]
            u = random.choice(comp_u)
            v = random.choice(comp_v)
            if (u, v) in edges:
                tries += 1
                if tries > 200:
                    break
                continue
            edges.append((u, v))
            inter_placed += 1
            tries = 0
            if inter_placed > max_inter_edges * 4:
                break

        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)

        sccs = sorted([sorted(c) for c in nx.strongly_connected_components(g)],
                      key=lambda c: c[0])
        gold_comps = kosaraju_scc(n, edges)
        assert [list(c) for c in sccs] == gold_comps, "internal SCC mismatch"

        ans_components = []
        if mode == "component":
            query = comps_nodes[0][0]
            component = next(c for c in sccs if query in c)
            answer = ",".join(str(x) for x in component)
        else:
            parts = ["[" + ",".join(str(x) for x in c) + "]" for c in sccs]
            answer = ";".join(parts)

        metadata = edict({
            "n": n,
            "edges": sorted(edges),
            "sccs": sccs,
            "mode": mode,
            "query": query if mode == "component" else None,
        })
        metadata.payload = {
            "n": n,
            "edges": sorted(edges),
            "mode": mode,
            "query": metadata.query,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        edge_list = ";".join(f"{u}->{v}" for u, v in metadata.edges)
        if metadata.mode == "component":
            return (f"Consider the directed graph on nodes 0..{metadata.n - 1} with "
                    f"edges: {edge_list}. Partition its nodes into strongly "
                    f"connected components (SCCs), using Kosaraju's algorithm, "
                    f"where a component is a maximal set of nodes mutually reachable "
                    f"from each other. Give the component that contains node "
                    f"{metadata.query}, with its nodes listed in increasing order "
                    f"separated by commas. The answer is that list, e.g. "
                    f"\"1,2,5\".")
        else:
            return (f"Consider the directed graph on nodes 0..{metadata.n - 1} with "
                    f"edges: {edge_list}. Partition its nodes into strongly "
                    f"connected components (SCCs), where a component is a maximal "
                    f"set of nodes mutually reachable from each other. Give the "
                    f"complete partition: each component as its nodes in increasing "
                    f"order inside square brackets, components ordered by their "
                    f"smallest node, and components separated by semicolons. For "
                    f"example \"[0,2];[1];[3,4]\". The answer is that string.")

    def score_answer(self, answer, entry):
        meta = entry.metadata
        gold = entry.answer
        a = answer.strip()
        if meta.mode == "component":
            gold_set = set(gold.split(","))
            try:
                ans_set = set("".join(a.split()).split(","))
            except Exception:
                return 0.0
            if ans_set == gold_set:
                return 1.0
            return 0.0
        else:
            try:
                cleaned = "".join(a.split())
            except Exception:
                return 0.0
            if cleaned == gold:
                return 1.0
            return 0.0
