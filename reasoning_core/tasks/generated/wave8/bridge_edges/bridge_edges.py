"""Given an undirected graph, output all bridge edges canonically."""

from dataclasses import dataclass

import networkx as nx
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'bridge_edges (draw 1 of 2)',
 'hypothesis': 'W1-003',
 'changes': 'new task in reasoning_core/tasks/generated/wave8/bridge_edges',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2592893641,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class BridgeEdgesConfig(Config):
    n_min_bridges: int = 2
    n_max_bridges: int = 6
    block_min: int = 3
    block_max: int = 5

    def apply_difficulty(self, level):
        self.n_min_bridges = sround(self.n_min_bridges + level)
        self.n_max_bridges = sround(self.n_max_bridges + level + 2)
        self.block_min = sround(self.block_min + level)
        self.block_max = sround(self.block_max + level)


def _canonical_bridges(graph):
    bridges = list(nx.bridges(graph))
    canon = sorted((min(u, v), max(u, v)) for u, v in bridges)
    return canon


def _bridges_str(canon):
    if not canon:
        return "none"
    return "; ".join(f"{u}-{v}" for u, v in canon)


class BridgeEdges:
    """Construct an undirected graph whose bridges are exactly the edges linking
    a linear chain of internally 2-edge-connected blocks."""

    def __init__(self):
        self.rng = random

    def generate(self, n_min_bridges, n_max_bridges, block_min, block_max):
        n_bridges = self.rng.randint(n_min_bridges, n_max_bridges)
        blocks = []
        for _ in range(n_bridges + 1):
            size = self.rng.randint(block_min, block_max)
            blocks.append(size)
        graph = nx.Graph()
        offset = 0
        cuts = []
        for size in blocks:
            nodes = list(range(offset, offset + size))
            offset += size
            cycle = nodes + [nodes[0]]
            for a, b in zip(cycle, cycle[1:]):
                graph.add_edge(a, b)
            for _ in range(size):
                if self.rng.random() < 0.7:
                    a = self.rng.choice(nodes)
                    b = self.rng.choice(nodes)
                    if a != b and not graph.has_edge(a, b):
                        graph.add_edge(a, b)
            cuts.append(nodes[0])
        for i in range(len(blocks) - 1):
            u = cuts[i]
            v = offset_start(blocks, i)
            graph.add_edge(u, v)
        canon = _canonical_bridges(graph)
        assert len(canon) == n_bridges
        assert set(canon) == set((min(a, b), max(a, b)) for a, b in graph.edges() if is_bridge_edge(graph, a, b))
        return graph, canon


def offset_start(blocks, i):
    return sum(blocks[: i + 1])


def is_bridge_edge(graph, a, b):
    return (a, b) in nx.bridges(graph) or (b, a) in nx.bridges(graph)


class BridgeEdgesTask(Task):
    task_name = "bridge_edges"
    summary = "Given an undirected graph, output all bridge edges canonically (sorted u<v, semicolon-separated, or 'none'), across varied node counts, block sizes, and bridge counts."
    config_cls = BridgeEdgesConfig

    def generate_entry(self):
        cfg = self.config
        gen = BridgeEdges()
        graph, canon = gen.generate(cfg.n_min_bridges, cfg.n_max_bridges,
                                    cfg.block_min, cfg.block_max)
        edges = sorted((int(min(u, v)), int(max(u, v))) for u, v in graph.edges())
        metadata = edict({
            "n": int(graph.number_of_nodes()),
            "m": int(graph.number_of_edges()),
            "bridges": _bridges_str(canon),
            "n_bridges": len(canon),
        })
        metadata.payload = {
            "nodes": [int(graph.number_of_nodes())],
            "edges": [[int(u), int(v)] for u, v in edges],
        }
        answer = _bridges_str(canon)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = [render_payload(metadata.payload)]
        lines.append(
            "The nodes are numbered 0 through %d (the label lists the %d nodes "
            "and %d edges). Name every bridge edge -- an edge whose removal "
            "disconnects the graph -- each as the two node numbers with the "
            "smaller first, e.g. 1-4. List all bridges separated by semicolons, "
            "in lexicographic order of the pairs." % (metadata.n - 1, metadata.n, metadata.m)
        )
        lines.append("The answer is the semicolon-separated list of bridges, or 'none' if there are none.")
        return "\n\n".join(lines)

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        answer = str(answer).strip()
        gold = entry.answer.strip()
        if answer == gold:
            return 1.0
        gold_set = _parse(gold)
        answer_set = _parse(answer)
        if gold_set is None or answer_set is None:
            return 0.0
        if answer_set == gold_set:
            return 1.0
        return 0.0


def _parse(astr):
    astr = astr.strip()
    if astr == "none":
        return set()
    pairs = astr.split(";")
    out = set()
    for p in pairs:
        p = p.strip()
        if not p:
            continue
        try:
            a, b = p.split("-")
            u, v = int(a.strip()), int(b.strip())
        except Exception:
            return None
        out.add((min(u, v), max(u, v)))
    return out
