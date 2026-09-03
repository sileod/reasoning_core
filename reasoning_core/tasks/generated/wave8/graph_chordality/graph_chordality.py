import random
import networkx as nx
from itertools import combinations
from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'graph_chordality (draw 1 of 2)',
 'hypothesis': 'W1-008',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/graph_chordality',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1705491944,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _longest_chordless_cycle(g):
    """Length of the longest induced (chordless) cycle of length >= 4,
    or 0 if the graph is chordal. An induced cycle of length L is a set of
    L vertices in which every vertex has exactly 2 neighbours within the set
    and there are exactly L edges among them."""
    nodes = sorted(g.nodes())
    n = len(nodes)
    best = 0
    for length in range(4, n + 1):
        for combo in combinations(nodes, length):
            sub = nx.induced_subgraph(g, combo)
            if nx.number_of_edges(sub) == length:
                if all(d == 2 for _, d in sub.degree()):
                    best = length
    return best


def _answer_string(g, lcc):
    if nx.is_chordal(g):
        return "true"
    return "false " + str(lcc)


def _chordal_blob(m, offset):
    """A connected chordal graph on m vertices labelled offset..offset+m-1,
    returned as a list of (u,v) edges (u<v), or [] if not found."""
    for _ in range(300):
        p = random.uniform(0.3, 0.9)
        g = nx.gnp_random_graph(m, p, seed=random.randrange(2**32))
        if nx.is_connected(g) and nx.is_chordal(g):
            return [(int(a), int(b)) for a, b in g.edges() if (int(a) < int(b))]
    return []


def _build_nonchordal(config):
    """Build a connected non-chordal graph whose longest chordless cycle is
    a chosen length L >= 4, constructed as an induced cycle C_L (the spine)
    with chordal blobs attached at articulation points so no longer chordless
    cycle appears. Returns (edge_list, lcc) or None."""
    n = config.n_nodes
    for _ in range(400):
        L = random.randint(4, n)
        n_blob = n - L
        edges = [(i, (i + 1) % L) for i in range(L)]
        offset = L
        used = 0
        if n_blob >= 1:
            targets = random.sample(range(L), k=min(n_blob, L))
            j = 0
            while used < n_blob:
                m = random.randint(1, n_blob - used)
                if m > 1:
                    blob_edges = _chordal_blob(m, offset)
                    for a, b in blob_edges:
                        edges.append((int(a), int(b)))
                edges.append((offset, targets[j % len(targets)]))
                offset += m
                used += m
                j += 1
        g = nx.Graph()
        g.add_edges_from(edges)
        if len(g.nodes()) != n:
            continue
        if nx.is_connected(g):
            lcc = _longest_chordless_cycle(g)
            if lcc == L:
                es = sorted((int(u), int(v)) for u, v in g.edges())
                return es, L
    return None


def _build_chordal(config):
    """A connected chordal graph, or None."""
    n = config.n_nodes
    for _ in range(2000):
        g = nx.gnp_random_graph(n, config.edge_prob, seed=random.randrange(2**32))
        if nx.is_connected(g) and nx.is_chordal(g):
            return g
    return None


class GraphChordalityConfig(Config):
    n_nodes: int = 7
    edge_prob: float = 0.55

    def apply_difficulty(self, level):
        self.n_nodes = min(13, sround(6 + level))
        self.edge_prob = min(0.9, 0.5 + 0.06 * level)


class GraphChordality(Task):
    summary = "Given an undirected graph, answer whether every cycle of length at least four has a chord."
    config_cls = GraphChordalityConfig

    def generate_entry(self):
        config = self.config
        if random.random() < 0.25:
            g = _build_chordal(config)
            if g is None:
                raise RuntimeError("failed to build a chordal graph")
            chordal = True
            lcc = 0
            edges_list = sorted((int(u), int(v)) for u, v in g.edges())
        else:
            res = _build_nonchordal(config)
            if res is None:
                raise RuntimeError("failed to build a non-chordal graph")
            edges_list, lcc = res
            g = nx.Graph()
            g.add_edges_from(edges_list)
            chordal = nx.is_chordal(g)
            if not (not chordal and 4 <= lcc <= config.n_nodes):
                raise RuntimeError("invalid non-chordal witness")
        answer = _answer_string(g, lcc)
        nodes = sorted(int(x) for x in (g.nodes() if g.number_of_nodes() else range(config.n_nodes)))
        if len(nodes) != config.n_nodes:
            raise RuntimeError("node count mismatch")
        metadata = edict({
            "nodes": nodes,
            "edges": edges_list,
            "chordal": bool(chordal),
            "longest_chordless_cycle": int(lcc),
        })
        metadata.payload = {
            "nodes": nodes,
            "edges": edges_list,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        n = len(metadata.payload["nodes"])
        edge_list = ", ".join(f"{u}-{v}" for u, v in metadata.payload["edges"])
        return (
            f"An undirected graph has vertices 0 through {n - 1}. "
            f"Its edges are: {edge_list}.\n\n"
            f"A chord in a cycle is an edge joining two nonconsecutive vertices of that cycle. "
            f"Answer whether every cycle of length at least four has a chord (that is, whether "
            f"the graph is chordal).\n\n"
            f"Answer with exactly 'true' if chordal, otherwise exactly 'false <L>' where L is the "
            f"length of the longest cycle of length at least four that has no chord.\n"
            f"Example format: 'true' or 'false 5'."
        )

    def score_answer(self, answer, entry):
        a = str(answer).strip()
        gt = entry.answer
        if a == gt:
            return 1.0
        if gt == "true":
            return 1.0 if a.lower() == "true" else 0.0
        parts = a.split()
        if len(parts) == 2 and parts[0].lower() == "false":
            try:
                if int(parts[1]) == entry.metadata["longest_chordless_cycle"]:
                    return 1.0
            except ValueError:
                pass
        return 0.0
