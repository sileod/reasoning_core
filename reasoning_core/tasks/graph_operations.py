import networkx as nx
import random
from reasoning_core.template import Task, DevTask, Entry, Config, render_payload, stochastic_rounding as sround
from reasoning_core.utils import parse_space_ints
from dataclasses import dataclass

# --- Configuration for All Graph Tasks ---
@dataclass
class GraphReasoningConfig(Config):
    num_nodes: int = 6  # Needs >= 5 to avoid issues with some generators/tasks
    no_solution_prob: float = 0.1
    return_to_start_prob: float = 0.1
    pairs_render_together_prob: float = 0.7
    weighted_prob: float = 0.25
    weight_min: int = 1
    weight_max: int = 9
    def apply_difficulty(self, level):
        # Keep structural growth, but leave some of the difficulty budget to the
        # path-length curriculum in GraphPathfinding instead of prompt bulk.
        self.num_nodes = sround(6 * 1.4 ** level)

_GRAPH_GENERATORS = [
    (nx.fast_gnp_random_graph, {'p': (0.15, 0.4)}),
    (nx.watts_strogatz_graph, {'k': (2, 4), 'p': (0.1, 0.3)}),
    (nx.barabasi_albert_graph, {'m': (1, 3)}),
    (nx.random_regular_graph, {'d': (2, 4)}), 
    (nx.random_labeled_tree, {}),
    (nx.powerlaw_cluster_graph, {'m': (1, 3), 'p': (0.1, 0.5)}),
    (nx.random_geometric_graph, {'radius': (0.3, 0.7)}),
]

_GRID_GENERATOR = (nx.grid_2d_graph, {'m': (3, 5), 'n': (3, 5)})


class BaseGraphTask:
    """Handles shared, flexible directed graph generation and rendering."""
    def __init__(self, config=None):
        super().__init__(config=config or GraphReasoningConfig())

    def _generate_graph(self):
        """Randomly selects a topology, generates a graph, and converts to a unified DiGraph."""
        num_nodes = self.config.num_nodes
        
        graph_generators = list(_GRAPH_GENERATORS)
        if self.config.level >= 1:
            graph_generators.append(_GRID_GENERATOR)

        for _ in range(15): # Try multiple times to get a valid graph
            gen_func, params_ranges = random.choice(graph_generators)
            params = {'n': num_nodes}
            try:
                for p_name, p_range in params_ranges.items():
                    if isinstance(p_range[0], float):
                        params[p_name] = random.uniform(*p_range)
                    else:
                        params[p_name] = random.randint(*p_range)
                
                # Generate base topology
                G_undirected = gen_func(**params)
                
                if G_undirected.number_of_nodes() == 0:
                    continue
                    
                G_undirected = nx.convert_node_labels_to_integers(G_undirected)
                
                # Unify to DiGraph
                G = G_undirected.to_directed()
                
                # Break symmetry to create true directed behavior for most graphs
                # ~80% of the time, selectively drop some reverse edges. 
                # Remaining ~20% are left fully reciprocal.
                if random.random() < 0.8:
                    edges_to_remove = []
                    for u, v in list(G.edges()):
                        if u < v and G.has_edge(v, u):
                            r = random.random()
                            if r < 0.33:
                                edges_to_remove.append((u, v)) # Drop forward
                            elif r < 0.66:
                                edges_to_remove.append((v, u)) # Drop reverse
                    G.remove_edges_from(edges_to_remove)

                # Prefer at least weakly connected graphs for structural tasks
                if nx.is_weakly_connected(G) and G.number_of_edges() > 0:
                    return G
            except (nx.NetworkXError, ValueError):
                continue 
        
        # Fallback if generators fail heavily
        G = nx.fast_gnp_random_graph(num_nodes, 0.4, directed=True)
        return nx.convert_node_labels_to_integers(G)

    def _format_edge(self, u, v, data, weighted):
        if weighted:
            return f"{u}->{v}({data['weight']})"
        return f"{u}->{v}"

    def _render_edges(self, G, weighted=False):
        return ", ".join(
            self._format_edge(u, v, d, weighted)
            for u, v, d in sorted(G.edges(data=True))
        )

    def _render_graph(self, G, weighted=False):
        """Randomly selects a method to describe the directed graph in text."""
        if weighted:
            return f"Nodes {sorted(list(G.nodes()))}. Directed Edges: {self._render_edges(G, weighted=True)}"

        def r_adjacency_list(g):
            return "\n".join(
                f"Node {n} has directed edges to: {', '.join(map(str, sorted(g.successors(n))))}."
                if g.out_degree(n) > 0 else f"Node {n} has no outgoing edges."
                for n in sorted(g.nodes())
            )

        def r_edge_list(g):
            edges_str = ", ".join(f"({u}, {v})" for u, v in sorted(list(g.edges())))
            return f"Nodes {sorted(list(g.nodes()))} and directed edges: {edges_str}."
        
        def r_adj_dict(g):
            return f"Adjacency Dictionary (source to targets): " + str({n: sorted(list(g.successors(n))) for n in sorted(g.nodes())})
        
        def r_edge_pairs(g):
            edges = [f"{u}->{v}" for u, v in sorted(g.edges())]
            return f"Directed Edges: {', '.join(edges)}"
        
        def r_adjacency_matrix(g):
            nodes = sorted(g.nodes())
            matrix = [[1 if g.has_edge(i, j) else 0 for j in nodes] for i in nodes]
            return f"Nodes: {nodes}\nAdjacency Matrix (row indicates source, column indicates target):\n" + "\n".join(map(str, matrix))
        
        def r_dot_notation(g):
            edges = "; ".join(f"{u}->{v}" for u, v in sorted(g.edges()))
            return f"digraph {{ {edges} }}"
        
        def r_prose(g):
            return " ".join(
                f"Node {n} points to {', '.join(map(str, sorted(g.successors(n))))}." 
                if g.out_degree(n) > 0 else f"Node {n} has no outgoing links."
                for n in sorted(g.nodes()))
        
        def r_incidence(g):
            return "; ".join(
                f"{n}: {' '.join(f'{n}->{nb}' for nb in sorted(g.successors(n)))}"
                if g.out_degree(n) > 0 else f"{n}:"
                for n in sorted(g.nodes()))

        renderers = [r_adjacency_list, r_edge_list, r_adj_dict, r_edge_pairs, r_adjacency_matrix, r_dot_notation, r_prose, r_incidence]
        return random.choice(renderers)(G)


class GraphPathfinding(BaseGraphTask, Task):
    summary = "Find the shortest path or cost in weighted and unweighted directed graphs."
    def _add_weights(self, G, weighted):
        for u, v in G.edges:
            G.edges[u, v]["weight"] = (
                random.randint(self.config.weight_min, self.config.weight_max)
                if weighted else 1
            )
        return G

    def _shortest_path(self, G, start, end):
        import heapq

        heap = [(0, (start,), start)]
        best = {}

        while heap:
            cost, path, u = heapq.heappop(heap)

            if u in best and best[u] <= (cost, path):
                continue
            best[u] = (cost, path)

            if u == end:
                return list(path), cost

            for v in sorted(G.successors(u)):
                w = G.edges[u, v].get("weight", 1)
                heapq.heappush(heap, (cost + w, path + (v,), v))

        return None, None

    def _has_optimal_tie(self, G, start, end):
        distances = nx.single_source_dijkstra_path_length(G, start, weight="weight")
        ways = {start: 1}
        for u in sorted(distances, key=lambda node: (distances[node], node)):
            for v in G.successors(u):
                weight = G.edges[u, v].get("weight", 1)
                if distances.get(v) == distances[u] + weight:
                    ways[v] = min(2, ways.get(v, 0) + ways.get(u, 0))
        return ways.get(end, 0) > 1

    def _disconnected_graph(self):
        n = max(2, self.config.num_nodes)
        n1 = random.randint(1, n - 1)
        G1 = nx.fast_gnp_random_graph(n1, 0.5, directed=True)
        G2 = nx.fast_gnp_random_graph(n - n1, 0.5, directed=True)
        return nx.disjoint_union(G1, G2)

    def _reachable_pair(self, G):
        pairs = [
            (u, v, hops)
            for u, distances in nx.all_pairs_shortest_path_length(G)
            for v, hops in distances.items()
            if u != v
        ]
        if not pairs:
            return None
        # L0 remains uniform. Higher levels increasingly prefer longer genuine
        # shortest paths without imposing a brittle minimum or retry loop.
        exponent = self.config.level / 2
        return random.choices(
            pairs, weights=[hops ** exponent for _, _, hops in pairs], k=1
        )[0]

    def make_cot(self, G, start, end, path=None, cost=None):
        if path is None and cost is None:
            path, cost = self._shortest_path(G, start, end)
        if path is None:
            return f"No directed path from {start} to {end}."
        return f"Optimal cost: {cost}. Path: {' '.join(map(str, path))}."

    def generate_entry(self):
        weighted = random.random() < self.config.weighted_prob
        G = self._generate_graph()
        selected_hops = None
        
        if random.random() < self.config.no_solution_prob:
            pairs = [(u, v) for u in G.nodes() for v in G.nodes() if u != v and not nx.has_path(G, u, v)]
            if not pairs:
                G = self._disconnected_graph()
                nodes1 = list(nx.weakly_connected_components(G))[0]
                nodes2 = list(set(G.nodes()) - nodes1)
                start, end = random.choice(list(nodes1)), random.choice(list(nodes2))
            else:
                start, end = random.choice(pairs)
            G = self._add_weights(G, weighted)
            path, cost = None, None
        else:
            pair = self._reachable_pair(G)
            if pair is None:
                G = nx.path_graph(self.config.num_nodes, create_using=nx.DiGraph)
                start, end = 0, self.config.num_nodes - 1
                selected_hops = self.config.num_nodes - 1
            else:
                start, end, selected_hops = pair
            G = self._add_weights(G, weighted)
            path, cost = self._shortest_path(G, start, end)

        graph_description = self._render_graph(G, weighted=weighted)
        mention_no_path = path is None or random.random() < 0.5
        mention_ties = (
            path is not None
            and (self._has_optimal_tie(G, start, end) or random.random() < 0.5)
        )
        return Entry(
            metadata={
                "weighted": weighted,
                "mention_no_path": mention_no_path,
                "mention_ties": mention_ties,
                "graph_description": graph_description, "start_node": start, "end_node": end,
                "payload": {"graph": graph_description},
                "nodes": list(G.nodes()), "edges": [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
                "optimal_cost": cost,
                "optimal_length": len(path) if path is not None else None,
                "selected_hops": selected_hops if path is not None else None,
                "cot": self.make_cot(G, start, end, path, cost)
            },
            answer="None" if path is None else " ".join(map(str, path))
        )

    def render_prompt(self, m):
        objective = "minimum-cost" if m.get("weighted") else "shortest"
        answer_format = "Return space-separated nodes"
        if m.get("mention_no_path", True):
            answer_format += ", or `None` if no path exists"
        tie_rule = "Break ties lexicographically. " if m.get("mention_ties", True) else ""
        return (
            f"Find the {objective} directed path from node {m['start_node']} "
            f"to node {m['end_node']}. {tie_rule}"
            f"{answer_format}.\n\n"
            f"{render_payload(m['payload'])}"
        )

    def score_answer(self, answer, entry):
            text = str(answer).strip()
            if "none" in text.lower():
                pred = None
            else:
                pred = parse_space_ints(text)

            meta = entry.metadata
            opt = meta.get("optimal_cost")
            legacy_opt_len = meta.get("optimal_length")
            target = opt if opt is not None else legacy_opt_len
            if pred is None: return 1.0 if target is None else 0.0
            if not isinstance(pred, list) or not pred: return 0.0

            th = lambda x: tuple(x) if isinstance(x, list) else x
            
            # --- Backwards Compatibility Fix ---
            # Look at the description to detect if the data was generated by V2 (directed) or V1 (undirected).
            # V2 descriptions always include one of these specific keywords/symbols.
            desc = meta.get("graph_description", "").lower()
            is_directed = any(kw in desc for kw in ["directed", "digraph", "source", "points", "->"])
            
            G = nx.DiGraph() if is_directed else nx.Graph()
            # -----------------------------------

            G.add_nodes_from(map(th, meta["nodes"]))
            for edge in meta["edges"]:
                if len(edge) == 3:
                    u, v, w = edge
                else:
                    u, v = edge
                    w = 1
                G.add_edge(th(u), th(v), weight=w)

            pred = list(map(th, pred))
            if pred[0] != th(meta["start_node"]) or pred[-1] != th(meta["end_node"]): return 0.0
            if not nx.is_path(G, pred) or target is None: return 0.0
            
            cost = sum(G.edges[u, v]["weight"] for u, v in zip(pred, pred[1:])) if opt is not None else len(pred)
            if cost < target:
                return 0.0
            return target / cost


@dataclass
class GraphSuccessorsConfig(Config):
    num_nodes: int = 6
    num_queries: int = 1
    max_hops: int = 2

    def apply_difficulty(self, level):
        self.num_nodes += level
        self.num_queries = sround(self.num_queries + 0.5 * level)
        self.max_hops += level


class GraphSuccessors(BaseGraphTask, Task):
    """DEPO-style k-th successor queries in a permutation digraph."""
    summary = "Determine the k-th successor of a node in a permutation digraph topology."
    def __init__(self, config=None):
        super().__init__(config=config or GraphSuccessorsConfig())

    def _jump(self, succ, x, k):
        for _ in range(k):
            x = succ[x]
        return x

    def generate_entry(self):
        nodes = list(range(self.config.num_nodes))
        succ = dict(zip(nodes, random.sample(nodes, len(nodes))))  # Ensure exact out-degree 1 per node

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(succ.items())

        queries = [
            (random.choice(nodes), random.randint(1, self.config.max_hops))
            for _ in range(self.config.num_queries)
        ]
        answer = [self._jump(succ, x, k) for x, k in queries]

        graph_description = self._render_graph(G)
        return Entry(
            metadata={
                "graph_description": graph_description,
                "queries": queries,
                "payload": {"graph": graph_description, "queries": str(queries)},
                "nodes": nodes,
                "edges": list(G.edges()),
            },
            answer=" ".join(map(str, answer)),
        )

    def render_prompt(self, m):
        return (
            "For each query (x, k), give the k-th successor of x by following directed edges k times.\n"
            "Answer with space-separated integers in query order.\n\n"
            f"{render_payload(m['payload'])}"
        )

    def score_answer(self, answer, entry):
        pred = parse_space_ints(answer)
        true = parse_space_ints(entry.answer)
        if pred is None or true is None or len(pred) != len(true):
            return 0.0
        return sum(p == t for p, t in zip(pred, true)) / len(true)


@dataclass
class GraphDependenciesConfig(Config):
    num_nodes: int = 6
    max_prereqs: int = 2

    def apply_difficulty(self, level):
        self.num_nodes += level
        self.max_prereqs = sround(self.max_prereqs + 0.5 * level)


class GraphDependencies(BaseGraphTask, DevTask):
    """BREVO-style recursive dependency resolution implemented via DAG topologies."""
    summary = "Resolve recursive node prerequisites in directed acyclic graphs (DAGs)."
    def __init__(self, config=None):
        super().__init__(config=config or GraphDependenciesConfig())

    def _make_dag(self):
        for _ in range(10):
            G = self._generate_graph()
            # Randomize order to safely drop reverse edges and create a DAG
            order = {n: i for i, n in enumerate(random.sample(list(G.nodes()), G.number_of_nodes()))}
            edges_to_remove = [(u, v) for u, v in G.edges() if order[u] >= order[v]]
            G.remove_edges_from(edges_to_remove)
            if G.number_of_edges() > 0:
                return G
                
        # Safe fallback
        G = nx.DiGraph()
        G.add_edges_from([(i, i+1) for i in range(self.config.num_nodes - 1)])
        return G

    def generate_entry(self):
        for _ in range(100):
            G = self._make_dag()
            # Find candidate that has at least two prerequisites to trace
            candidates = [u for u in G.nodes() if len(nx.ancestors(G, u)) >= 2]
            if not candidates:
                continue

            q = random.choice(candidates)
            need = nx.ancestors(G, q)

            # Standard topological sort places ancestors (prerequisites) first
            answer = list(nx.lexicographical_topological_sort(G.subgraph(need)))

            graph_description = self._render_graph(G)
            return Entry(
                metadata={
                    "graph_description": graph_description,
                    "query": q,
                    "payload": {"graph": graph_description},
                    "nodes": list(G.nodes()),
                    "edges": list(G.edges()),
                },
                answer=" ".join(map(str, answer)),
            )
        return self.generate_entry()

    def render_prompt(self, m):
        return (
            f"List all ancestors of node {m['query']}.\n"
            "Order them so predecessors come before successors, with lexicographic tie-breaks.\n"
            "Answer with space-separated indexes.\n\n"
            f"{render_payload(m['payload'])}"
        )

    def score_answer(self, answer, entry):
        pred = parse_space_ints(answer)
        if pred is None:
            return 0.0

        m = entry.metadata
        G = nx.DiGraph()
        G.add_nodes_from(m["nodes"])
        G.add_edges_from(m["edges"])

        need = nx.ancestors(G, m["query"])
        if len(pred) != len(need) or set(pred) != need:
            return 0.0

        pos = {x: i for i, x in enumerate(pred)}
        for u, v in G.subgraph(need).edges():
            # edge u->v means u is prereq of v, so u must come before v
            if pos[u] > pos[v]:  
                return 0.0
        return 1.0
