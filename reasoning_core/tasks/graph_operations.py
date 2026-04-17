import networkx as nx
import random
from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass
from ast import literal_eval

# --- Configuration for All Graph Tasks ---
@dataclass
class GraphReasoningConfig(Config):
    num_nodes: int = 5  #needs 5 to avoid duplicates
    no_solution_prob: float = 0.1
    return_to_start_prob: float = 0.1

    def update(self, c): 
        self.num_nodes *= (1+c)


_GRAPH_GENERATORS = [
    (nx.fast_gnp_random_graph, {'p': (0.15, 0.4)}),
    (nx.watts_strogatz_graph, {'k': (2, 4), 'p': (0.1, 0.3)}),
    (nx.barabasi_albert_graph, {'m': (1, 3)}),
    (nx.random_regular_graph, {'d': (2, 4)}), # Every node has d neighbors

    # NB: we avoid graphs generating too much repetitions
    (nx.random_labeled_tree, {}),
    (nx.powerlaw_cluster_graph, {'m': (1, 3), 'p': (0.1, 0.5)}),
    (nx.random_geometric_graph, {'radius': (0.3, 0.7)}),
]

# Keep grid generation for higher levels only to avoid tuple-node complexity at level 0.
_GRID_GENERATOR = (nx.grid_2d_graph, {'m': (3, 5), 'n': (3, 5)})


class BaseGraphTask:
    """Handles shared, flexible graph generation and rendering."""
    def __init__(self, config=GraphReasoningConfig()):
        super().__init__(config)

    def _generate_graph(self):
            """Randomly selects a topology from the list and generates a graph."""
            num_nodes = self.config.num_nodes
            
            graph_generators = list(_GRAPH_GENERATORS)
            if self.config.level >= 1:
                graph_generators.append(_GRID_GENERATOR)

            for _ in range(10): # Try a few times to get a valid graph
                gen_func, params_ranges = random.choice(graph_generators)
                params = {'n': num_nodes}
                try:
                    for p_name, p_range in params_ranges.items():
                        if isinstance(p_range[0], float):
                            params[p_name] = random.uniform(*p_range)
                        else:
                            params[p_name] = random.randint(*p_range)
                    
                    G = gen_func(**params)
                    # Ensure it's connected and has nodes for most tasks
                    if G.number_of_nodes() > 0 and nx.is_connected(G):
                        # This prevents nx.grid_2d_graph from returning tuple nodes.
                        return nx.convert_node_labels_to_integers(G)
                except (nx.NetworkXError, ValueError) as e:
                    continue # Some generators can fail with certain params, just retry
            
            return nx.convert_node_labels_to_integers(nx.fast_gnp_random_graph(num_nodes, 0.5))

    def _render_graph(self, G):
        """Randomly selects a method to describe the graph in text."""
        def r_adjacency_list(g):
            return "\n".join(
                f"Node {n} is connected to: {', '.join(map(str, sorted(g.neighbors(n))))}."
                for n in sorted(g.nodes())
            )

        def r_edge_list(g):
            edges_str = ", ".join(map(str, sorted(list(g.edges()))))
            return f"Nodes {sorted(list(g.nodes()))} and edges: {edges_str}."
        
        def r_adj_dict(g):
            return str({n: sorted(list(g.neighbors(n))) for n in sorted(g.nodes())})
        
        def r_edge_pairs(g):
            edges = [f"{u}-{v}" for u, v in sorted(g.edges())]
            return f"Edges: {', '.join(edges)}"
        
        def r_adjacency_matrix(g):
            nodes = sorted(g.nodes())
            matrix = [[1 if g.has_edge(i, j) else 0 for j in nodes] for i in nodes]
            return f"Nodes: {nodes}\nMatrix:\n" + "\n".join(map(str, matrix))
        
        def r_dot_notation(g):
            edges = "; ".join(f"{u}--{v}" for u, v in sorted(g.edges()))
            return f"graph {{ {edges} }}"
        
        def r_prose(g):
            return " ".join(
                f"Node {n} connects to {', '.join(map(str, sorted(g.neighbors(n))))}." 
                if g.degree(n) > 0 else f"Node {n} is isolated."
                for n in sorted(g.nodes()))
        
        def r_incidence(g):
            """Each node lists its edges"""
            return "; ".join(
                f"{n}: {' '.join(f'{n}-{nb}' for nb in sorted(g.neighbors(n)))}"
                for n in sorted(g.nodes()))
        renderers = [r_adjacency_list, r_edge_list, r_adj_dict, r_edge_pairs, r_adjacency_matrix, r_dot_notation, r_prose, r_incidence]
        renderer = random.choice(renderers)
        return renderer(G)


class GraphPathfinding(BaseGraphTask, Task):
    def _lexicographic_shortest_path(self, G, start, end):
        """BFS exploring neighbors in sorted order → lexicographically smallest shortest path."""
        queue = [(start, [start])]
        visited = {start}
        while queue:
            curr, path = queue.pop(0)
            if curr == end:
                return path
            for n in sorted(G.neighbors(curr)):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, path + [n]))
        return None

    def _disconnected_graph(self):
        """Build two connected components to guarantee unreachable pairs."""
        def connected_graph(n):
            for _ in range(20):
                G = nx.fast_gnp_random_graph(n, random.uniform(0.25, 0.75))
                if G.number_of_nodes() > 0 and nx.is_connected(G):
                    return G
            return nx.path_graph(n)

        n = max(2, self.config.num_nodes)
        n1 = random.randint(1, n - 1)
        g1 = connected_graph(n1)
        g2 = connected_graph(n - n1)
        return nx.disjoint_union(g1, g2)

    def _lexicographic_shortest_non_empty_cycle(self, G, start):
        candidates = []
        for n in sorted(G.neighbors(start)):
            H = G.copy()
            H.remove_edge(start, n)
            tail = self._lexicographic_shortest_path(H, n, start)
            if tail is not None:
                candidates.append([start] + tail)
        return min(candidates, key=lambda p: (len(p), p)) if candidates else None

    def _cycle_start_nodes(self, G):
        return [
            u for u in G.nodes()
            if self._lexicographic_shortest_non_empty_cycle(G, u) is not None
        ]

    def _sample_same_node_cycle(self, G):
        cycle_nodes = self._cycle_start_nodes(G)
        if not cycle_nodes:
            non_edges = list(nx.non_edges(G))
            if non_edges:
                G.add_edge(*random.choice(non_edges))
                cycle_nodes = self._cycle_start_nodes(G)
        if not cycle_nodes:
            return None, None
        start = random.choice(cycle_nodes)
        return start, self._lexicographic_shortest_non_empty_cycle(G, start)

    def make_cot(self, G, start, end):
        # BFS State Initialization
        queue = [(start, [start])] # Tuple: (Current Node, Path History)
        visited = {start}
        
        lines = [f"Goal: Shortest path from {start} to {end} using BFS."]
        lines.append(f"Initialize Queue: [{start}]")
        
        while queue:
            curr, path = queue.pop(0)
            lines.append(f"\nPop {curr}. Current Path: {path}")
            
            if curr == end:
                lines.append(f"Target {end} found! Search Complete.")
                return "\n".join(lines)
            
            # Explore Neighbors (Sorted for deterministic reasoning)
            new_neighbors = []
            for n in sorted(G.neighbors(curr)):
                if n not in visited:
                    visited.add(n)
                    new_neighbors.append(n)
                    queue.append((n, path + [n]))
            
            # Reasoning Step: Explain the update
            if new_neighbors:
                lines.append(f"  -> Found new neighbors: {new_neighbors}")
                lines.append(f"  -> Add to queue. Visited set updated.")
            else:
                lines.append(f"  -> All neighbors visited or empty. Backtrack.")
                
            # Explicit State Dump (Crucial for Transformer State Tracking)
            q_state = [n for n, _ in queue]
            lines.append(f"  -> Queue is now: {q_state}")

        return "Target unreachable."
    def generate(self):
        is_unsat = random.random() < self.config.no_solution_prob
        if is_unsat:
            G = self._disconnected_graph()
            cc = list(nx.connected_components(G))
            start = random.choice(list(cc[0]))
            end = random.choice(list(cc[1]))
            path = None
            return Problem(
                metadata={
                    "graph_description": self._render_graph(G),
                    "start_node": start,
                    "end_node": end,
                    "nodes": list(G.nodes()),
                    "edges": list(G.edges()),
                    "optimal_length": None,
                    "cot": self.make_cot(G, start, end),
                },
                answer="None",
            )

        G = self._generate_graph()
        if random.random() < self.config.return_to_start_prob:
            start, path = self._sample_same_node_cycle(G)
            if start is not None:
                end = start
            else:
                start, end = random.sample(list(G.nodes()), 2)
                path = self._lexicographic_shortest_path(G, start, end)
        else:
            start, end = random.sample(list(G.nodes()), 2)
            path = self._lexicographic_shortest_path(G, start, end)

        metadata = {
            "graph_description": self._render_graph(G), "start_node": start, "end_node": end,
            "nodes": list(G.nodes()), "edges": list(G.edges()),
            "optimal_length": len(path) if path is not None else None,
            "cot": self.make_cot(G, start, end)
        }
        return Problem(metadata=metadata, answer=str(path))

    def prompt(self, m):
        extra = " (non-empty)" if m["start_node"] == m["end_node"] else ""
        return (
            f"Consider the graph:\n\n{m['graph_description']}\n\n"
            f"Find the lexicographically smallest shortest path{extra} from Node {m['start_node']} to Node {m['end_node']}.\n"
            "If no path exists, answer `None`.\n"
            "The answer is a Python list of nodes or `None`."
        )

    def score_answer(self, answer, entry):
            try: pred_path = literal_eval(answer)
            except: return 0.0

            meta = entry.metadata
            if pred_path is None:
                return 1.0 if meta.get("optimal_length") is None else 0.0
            if not isinstance(pred_path, list) or len(pred_path) < 1:
                return 0.0
            
            def to_hashable(x):
                return tuple(x) if isinstance(x, list) else x

            nodes = [to_hashable(n) for n in meta['nodes']]
            edges = [(to_hashable(u), to_hashable(v)) for u, v in meta['edges']]
            
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            start_node = to_hashable(meta['start_node'])
            end_node = to_hashable(meta['end_node'])
            pred_path = [to_hashable(n) for n in pred_path]

            if (pred_path[0] != start_node or pred_path[-1] != end_node):
                return 0.0
            if start_node == end_node and len(pred_path) < 3:
                return 0.0
            if not nx.is_path(G, pred_path):
                return 0.0
            if meta['optimal_length'] is None:
                return 0.0
            return meta['optimal_length'] / len(pred_path)


class GraphNodeCentrality(BaseGraphTask, Task):
    """Task to find all nodes with the highest centrality in a graph."""

    def generate(self):
        G = self._generate_graph()
        
        # Degree centrality is simple and intuitive: node with most connections.
        centrality = nx.degree_centrality(G)
        if not centrality: # Handle empty graph case
            return self.generate()

        # Find the maximum centrality value.
        max_value = max(centrality.values())
        
        # Find all nodes that share this maximum value.
        most_central_nodes = sorted([
            node for node, value in centrality.items() if value == max_value
        ])
        
        metadata = {"graph_description": self._render_graph(G)}
        return Problem(metadata=metadata, answer=str(most_central_nodes))

    def prompt(self, metadata):
        return (
            f"Consider the following social network graph:\n\n{metadata['graph_description']}\n\n"
            "Based on the number of connections, identify all nodes that are the most central "
            "(i.e., have the highest degree centrality). There may be more than one.\n"
            "The answer is a Python list of node integers, sorted in increasing order. "
            "Example: `[3, 8]`."
        )

    def score_answer(self, answer, entry):
        """Scores based on whether the predicted list of nodes is exactly correct."""
        try:
            # Safely evaluate the string representations of the lists.
            pred_list = literal_eval(answer)
            true_list = literal_eval(entry.answer)
            # The lists must be identical (which also enforces the sorting rule).
            return 1.0 if pred_list == true_list else 0.0
        except:
            return 0.0

class GraphIsomorphism(BaseGraphTask, Task): 
    """Task to determine if two graphs have the exact same structure."""

    def generate(self):
        G1 = self._generate_graph()
        
        # We want False ~70% of the time.
        if random.random() < 0.3:
            # TRUE Case: Create a structurally identical graph by relabeling nodes.
            nodes = list(G1.nodes())
            mapping = dict(zip(nodes, random.sample(nodes, len(nodes))))
            G2 = nx.relabel_nodes(G1, mapping)
            answer = True
        else:
            G2 = G1.copy()
            success = False
            
            for _ in range(10):
                swaps = max(1, G2.number_of_edges() // 5)
                try:
                    nx.double_edge_swap(G2, nswap=swaps, max_tries=100)
                    if not nx.is_isomorphic(G1, G2):
                        success = True
                        break
                except nx.NetworkXError:  # Can fail on certain graph types
                    continue
            
            # Fallback: generate a completely different graph
            if not success:
                for _ in range(50):  # Prevent infinite loop
                    G2 = self._generate_graph()
                    if (G2.number_of_nodes() == G1.number_of_nodes() and 
                        not nx.is_isomorphic(G1, G2)):
                        break            
            answer = False  # ← Now INSIDE the else block

        metadata = {
            "graph1_description": self._render_graph(G1),
            "graph2_description": self._render_graph(G2),
        }
        return Problem(metadata=metadata, answer=str(answer))

    def prompt(self, metadata):
        return (
            f"Consider two graphs described below.\n\nGraph A:\n{metadata['graph1_description']}\n\n"
            f"Graph B:\n{metadata['graph2_description']}\n\n"
            "Do Graph A and Graph B have the exact same structure, just with different node labels? "
            "(In other words, are they isomorphic?)\n"
            "The answer is `True` or `False`."
        )

    def score_answer(self, answer, entry):
        return 1.0 if str(answer).strip().lower() == entry.answer.lower() else 0.0



def _parse_list(x):
    try:
        x = literal_eval(x)
        return x if isinstance(x, list) else None
    except Exception:
        return None


class BaseDirectedGraphTask:
    """Tiny helper for directed-graph rendering."""
    def _render_digraph(self, G):
        if random.random() < 0.5:
            edges = [f"{u}->{v}" for u, v in G.edges()]
            random.shuffle(edges)
            return "Edges: " + ", ".join(edges)
        return str({u: sorted(G.successors(u)) for u in sorted(G.nodes())})

    def _render_dependency_graph(self, G):
        """Render with explicit 'depends on' language — no room for misreading.

        Edge convention: prerequisite -> dependent  (u->v means v depends on u).
        So a node's *predecessors* are its prerequisites.
        """
        renderers = [
            # "X depends on: Y, Z"
            lambda g: "\n".join(
                f"Node {u} depends on: {', '.join(map(str, sorted(g.predecessors(u))))}."
                if g.in_degree(u) > 0 else f"Node {u} has no dependencies."
                for u in sorted(g.nodes())
            ),
            # dict labeled explicitly
            lambda g: "Dependencies (each key lists its prerequisites): " + str(
                {u: sorted(g.predecessors(u)) for u in sorted(g.nodes())}
            ),
            # arrow format with inline gloss
            lambda g: (
                "Edges (X->Y means Y depends on X): "
                + ", ".join(f"{u}->{v}" for u, v in sorted(g.edges()))
            ),
        ]
        return random.choice(renderers)(G)


# DEPO-style: repeated successor chasing in a permutation digraph
@dataclass
class GraphSuccessorsConfig(Config):
    num_nodes: int = 6
    num_queries: int = 1
    max_hops: int = 2

    def update(self, c=1):
        self.num_nodes += c
        self.num_queries += c / 2
        self.max_hops += c


class GraphSuccessors(BaseDirectedGraphTask, Task):
    """DEPO-style k-th successor queries."""
    def __init__(self, config=GraphSuccessorsConfig()):
        super().__init__(config=config)

    def _jump(self, succ, x, k):
        for _ in range(k):
            x = succ[x]
        return x

    def generate(self):
        nodes = list(range(self.config.num_nodes))
        succ = dict(zip(nodes, random.sample(nodes, len(nodes))))  # permutation

        G = nx.DiGraph()
        G.add_edges_from(succ.items())

        queries = [
            (random.choice(nodes), random.randint(1, self.config.max_hops))
            for _ in range(self.config.num_queries)
        ]
        answer = [self._jump(succ, x, k) for x, k in queries]

        return Problem(
            metadata={
                "graph_description": self._render_digraph(G),
                "queries": queries,
                "nodes": nodes,
                "edges": list(G.edges()),
            },
            answer=str(answer),
        )

    def prompt(self, m):
        return (
            f"Consider the directed graph:\n\n{m['graph_description']}\n\n"
            f"Queries: {m['queries']}\n"
            "Each pair (x, k) asks for the k-th successor of x.\n"
            "The answer is a Python list of integers in query order."
        )

    def score_answer(self, answer, entry):
        pred = _parse_list(answer)
        true = _parse_list(entry.answer)
        if pred is None or true is None or len(pred) != len(true):
            return 0.0
        return sum(p == t for p, t in zip(pred, true)) / len(true)


# BREVO-style: dependency expansion in a DAG
@dataclass
class GraphDependenciesConfig(Config):
    # Smaller defaults keep dependency prompts less compositional.
    num_nodes: int = 6
    max_prereqs: int = 2

    def update(self, c=1):
        self.num_nodes += c
        self.max_prereqs += c / 2


class GraphDependencies(BaseDirectedGraphTask, Task):
    """BREVO-style recursive dependency resolution."""
    def __init__(self, config=GraphDependenciesConfig()):
        super().__init__(config=config)

    def _make_dag(self):
        order = list(range(self.config.num_nodes))
        random.shuffle(order)

        G = nx.DiGraph()
        G.add_nodes_from(order)

        # Edge convention: prerequisite -> dependent
        # order[i] can only depend on earlier items order[:i]
        for i, v in enumerate(order):
            prev = order[:i]
            k = random.randint(0, min(len(prev), self.config.max_prereqs))
            for u in random.sample(prev, k):
                G.add_edge(u, v)  # u is prerequisite of v
        return G

    def generate(self):
        for _ in range(100):
            G = self._make_dag()
            candidates = [u for u in G.nodes() if len(nx.ancestors(G, u)) >= 2]
            if not candidates:
                continue

            q = random.choice(candidates)
            need = nx.ancestors(G, q)

            # leaves first among recursive dependents
            answer = list(
                nx.lexicographical_topological_sort(
                    G.subgraph(need)
                )
            )

            return Problem(
                metadata={
                    "graph_description": self._render_dependency_graph(G),
                    "query": q,
                    "nodes": list(G.nodes()),
                    "edges": list(G.edges()),
                },
                answer=str(answer),
            )
        return self.generate()

    def prompt(self, m):
        return (
            f"Consider the dependency graph:\n\n{m['graph_description']}\n\n"
            f"List all prerequisites of node {m['query']} (recursively), leaves first.\n"
            "Do not include the query node itself.\n"
            "If A depends on B and both appear in your answer, B must appear before A.\n"
            "The answer is a Python list of integers."
        )

    def score_answer(self, answer, entry):
        pred = _parse_list(answer)
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
            if pos[u] > pos[v]:  # edge u->v means u is prereq of v; u must come first
                return 0.0
        return 1.0
