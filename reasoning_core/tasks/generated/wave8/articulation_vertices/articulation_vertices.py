import random
from dataclasses import dataclass, field

import networkx as nx

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'articulation_vertices (draw 1 of 2)',
 'hypothesis': 'W1-002',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/articulation_vertices',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3438477119,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ArtVerticesConfig(Config):
    n_nodes: int = 7
    extra_scale: float = 0.4

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + level * 3)
        self.extra_scale = min(1.3, 0.4 + 0.15 * level)


def _render_graph(edges, n_nodes):
    lines = [f"n = {n_nodes}"]
    lines.append("edges = " + repr(sorted(list(edges))))
    return "\n".join(lines)


class ArticulationVertices(Task):
    summary = ("Given an undirected connected graph as a vertex count and edge list, "
               "find all articulation vertices and output them in ascending order, "
               "with empty output for graphs with no articulation vertices.")
    config_cls = ArtVerticesConfig

    def generate_entry(self):
        cfg = self.config
        while True:
            n = cfg.n_nodes
            possible = n * (n - 1) // 2
            # Build a random tree first so the graph is connected and each example
            # has a variety of articulation vertices before extra edges mask them.
            vertices = list(range(n))
            tree_edges = []
            for v in range(1, n):
                tree_edges.append((v, random.randrange(v)))
            max_m = n - 1 + int(round((n - 1) * cfg.extra_scale))
            if max_m > possible:
                max_m = possible
            m_target = random.randrange(n - 1, max_m + 1)
            rest = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(rest)
            extra = []
            for e in rest:
                if len(extra) >= m_target - (n - 1):
                    break
                if e not in tree_edges and (e[1], e[0]) not in tree_edges:
                    extra.append(e)
            edges = tree_edges + extra
            G = nx.Graph()
            G.add_nodes_from(vertices)
            G.add_edges_from(edges)
            arts = list(nx.articulation_points(G))
            if cmp := _check_articulations(G, set(arts)):
                raise RuntimeError(f"verifier mismatch: {cmp}")
            arts.sort()
            break
        answer = "[" + ", ".join(map(str, arts)) + "]"
        adj = []
        for v in vertices:
            adj.append(sorted(G.neighbors(v)))
        metadata = edict({
            "n_nodes": n,
            "edges": sorted(edges),
            "adjacency": adj,
            "payload": {
                "n": n,
                "edges": sorted(list(edges)),
            },
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = [
            f"We have an undirected graph with n = {metadata.n_nodes} vertices numbered 0 through "
            f"{metadata.n_nodes - 1}. Its edges are:",
            repr(metadata.edges),
            "",
            "A vertex is an articulation vertex if removing it (and all edges incident to it) "
            "increases the number of connected components of the graph.",
            "",
            "Output all articulation vertices of this graph, in ascending order of their labels, "
            "as the answer. If the graph has no articulation vertices, the answer is an empty list.",
            "",
            "Use standard depth-first-search articulation-point finding to determine the answer.",
        ]
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        import ast
        gt = ast.literal_eval(entry.answer)
        try:
            got = ast.literal_eval(answer.strip())
        except Exception:
            return 0.0
        if not isinstance(got, list):
            return 0.0
        try:
            got = sorted(int(x) for x in got)
        except Exception:
            return 0.0
        if got != gt:
            return 0.0
        return 1.0

    def distractor_candidates(self, entry):
        arts = set(ast_literal_eval(entry.answer))
        n = entry.n_nodes
        allv = list(range(n))
        # every vertex --- common failure (guessing all are articulation)
        yield "[" + ", ".join(map(str, allv)) + "]"
        # neighbors of a vertex
        if entry.adjacency:
            for v in allv:
                yield "[" + ", ".join(map(str, sorted(entry.adjacency[v]))) + "]"


def ast_literal_eval(ans):
    import ast
    return ast.literal_eval(ans)


def _check_articulations(G, arts):
    """Return a mismatch string if the given art set is wrong, else None."""
    # Verify by direct removal on a small graph: connected components must increase.
    import itertools
    n = G.number_of_nodes()
    # complete verification: for each candidate articulation, removal increases components
    computed = set()
    for v in G.nodes():
        G2 = G.copy()
        G2.remove_node(v)
        comps = nx.number_connected_components(G2)
        base = nx.number_connected_components(G)
        if comps > base:
            computed.add(v)
    if computed != set(arts):
        return f"expected {sorted(computed)} got {sorted(arts)}"
    return None
