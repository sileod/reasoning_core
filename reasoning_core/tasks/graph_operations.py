import networkx as nx
import random
from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass
from ast import literal_eval
import re

# --- Configuration for All Graph Tasks ---
@dataclass
class GraphReasoningConfig(Config):
    num_nodes: int = 5  # Needs >= 5 to avoid issues with some generators/tasks
    no_solution_prob: float = 0.1
    return_to_start_prob: float = 0.1

    def update(self, c): 
        self.num_nodes *= (1 + c)

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


def _parse_list(x):
    try:
        x = literal_eval(x)
        return x if isinstance(x, list) else None
    except Exception:
        return None


class BaseGraphTask:
    """Handles shared, flexible directed graph generation and rendering."""
    def __init__(self, config=GraphReasoningConfig()):
        super().__init__(config)

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

    def _render_graph(self, G):
        """Randomly selects a method to describe the directed graph in text."""
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
    def _lexicographic_shortest_path(self, G, start, end):
        queue, visited = [(start, [start])], {start}
        while queue:
            curr, path = queue.pop(0)
            if curr == end: return path
            for n in sorted(G.successors(curr)):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, path + [n]))
        return None

    def _disconnected_graph(self):
        n = max(2, self.config.num_nodes)
        n1 = random.randint(1, n - 1)
        G1 = nx.fast_gnp_random_graph(n1, 0.5, directed=True)
        G2 = nx.fast_gnp_random_graph(n - n1, 0.5, directed=True)
        return nx.disjoint_union(G1, G2)

    def make_cot(self, G, start, end):
        queue, visited = [(start, [start])], {start}
        lines = [f"Goal: Shortest directed path from {start} to {end} using BFS.", f"Initialize Queue: [{start}]"]
        while queue:
            curr, path = queue.pop(0)
            lines.append(f"\nPop {curr}. Current Path: {path}")
            if curr == end:
                lines.append(f"Target {end} found! Search Complete.")
                return "\n".join(lines)
            
            new_successors = [n for n in sorted(G.successors(curr)) if n not in visited]
            for n in new_successors:
                visited.add(n)
                queue.append((n, path + [n]))
            
            if new_successors:
                lines.extend([f"  -> Found new outgoing neighbors: {new_successors}", "  -> Add to queue. Visited set updated."])
            else:
                lines.append("  -> All outgoing neighbors visited or empty. Backtrack.")
            lines.append(f"  -> Queue is now: {[n for n, _ in queue]}")
        return "\n".join(lines + ["Target unreachable."])

    def generate(self):
        G = self._generate_graph()
        
        if random.random() < self.config.no_solution_prob:
            pairs = [(u, v) for u in G.nodes() for v in G.nodes() if u != v and not nx.has_path(G, u, v)]
            if not pairs:
                G = self._disconnected_graph()
                nodes1 = list(nx.weakly_connected_components(G))[0]
                nodes2 = list(set(G.nodes()) - nodes1)
                start, end = random.choice(list(nodes1)), random.choice(list(nodes2))
            else:
                start, end = random.choice(pairs)
            path = None
        else:
            pairs = [(u, v) for u in G.nodes() for v in G.nodes() if u != v and nx.has_path(G, u, v)]
            if not pairs:
                G = nx.path_graph(self.config.num_nodes, create_using=nx.DiGraph)
                start, end = 0, self.config.num_nodes - 1
            else:
                start, end = random.choice(pairs)
            path = self._lexicographic_shortest_path(G, start, end)

        return Problem(
            metadata={
                "graph_description": self._render_graph(G), "start_node": start, "end_node": end,
                "nodes": list(G.nodes()), "edges": list(G.edges()),
                "optimal_length": len(path) if path is not None else None,
                "cot": self.make_cot(G, start, end)
            },
            answer=str(path)
        )

    def prompt(self, m):
        return (
            f"Consider the directed graph:\n\n{m['graph_description']}\n\n"
            f"Find the lexicographically smallest shortest directed path from Node {m['start_node']} to Node {m['end_node']}.\n"
            "If no path exists, answer `None`.\n"
            "The answer is a Python list of nodes or `None`."
        )

    def score_answer(self, answer, entry):
            text = str(answer).strip()
            if "none" in text.lower():
                pred = None
            else:
                try: pred = literal_eval(text)
                except Exception:
                    m = re.search(r"\[[^\]]*\]", text)
                    try: pred = literal_eval(m.group(0)) if m else None
                    except Exception: return 0.0

            meta, opt_len = entry.metadata, entry.metadata.get("optimal_length")
            if pred is None: return 1.0 if opt_len is None else 0.0
            if isinstance(pred, tuple): pred = list(pred)
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
            G.add_edges_from((th(u), th(v)) for u, v in meta["edges"])

            pred = list(map(th, pred))
            if pred[0] != th(meta["start_node"]) or pred[-1] != th(meta["end_node"]): return 0.0
            if not nx.is_path(G, pred) or opt_len is None or len(pred) < opt_len: return 0.0
            
            return opt_len / len(pred)

class GraphNodeCentrality(BaseGraphTask, Task):
    """Task to find all nodes with the highest degree centrality in a directed graph."""

    def generate(self):
        G = self._generate_graph()
        
        # Centrality is evaluated as total degree (incoming + outgoing connections)
        degrees = dict(G.degree())
        if not degrees:
            return self.generate()

        max_degree = max(degrees.values())
        most_central_nodes = sorted([node for node, deg in degrees.items() if deg == max_degree])
        
        metadata = {"graph_description": self._render_graph(G)}
        return Problem(metadata=metadata, answer=str(most_central_nodes))

    def prompt(self, metadata):
        return (
            f"Consider the following directed network graph:\n\n{metadata['graph_description']}\n\n"
            "Based on the total number of connections (summing both incoming and outgoing edges for each node), "
            "identify all nodes that are the most central (i.e., have the highest total degree).\n"
            "There may be more than one.\n"
            "The answer is a Python list of node integers, sorted in increasing order. "
            "Example: `[3, 8]`."
        )

    def score_answer(self, answer, entry):
        try:
            pred_list = literal_eval(answer)
            true_list = literal_eval(entry.answer)
            return 1.0 if pred_list == true_list else 0.0
        except:
            return 0.0


class GraphIsomorphism(BaseGraphTask, Task): 
    """Task to determine if two directed graphs have the exact same structure."""

    def generate(self):
        G1 = self._generate_graph()
        
        if random.random() < 0.3:
            # TRUE Case
            nodes = list(G1.nodes())
            mapping = dict(zip(nodes, random.sample(nodes, len(nodes))))
            G2 = nx.relabel_nodes(G1, mapping)
            answer = True
        else:
            # FALSE Case
            G2 = G1.copy()
            success = False
            for _ in range(15):
                edges = list(G2.edges())
                if len(edges) >= 2:
                    e1, e2 = random.sample(edges, 2)
                    # Manually rewire to break isomorphism
                    if e1[0] != e2[1] and e2[0] != e1[1]:
                        G2.remove_edge(*e1)
                        G2.remove_edge(*e2)
                        G2.add_edge(e1[0], e2[1])
                        G2.add_edge(e2[0], e1[1])
                        if not nx.is_isomorphic(G1, G2):
                            success = True
                            break
                            
            if not success:
                for _ in range(25):
                    G2 = self._generate_graph()
                    if (G2.number_of_nodes() == G1.number_of_nodes() and 
                        not nx.is_isomorphic(G1, G2)):
                        break            
            answer = False

        metadata = {
            "graph1_description": self._render_graph(G1),
            "graph2_description": self._render_graph(G2),
        }
        return Problem(metadata=metadata, answer=str(answer))

    def prompt(self, metadata):
        return (
            f"Consider two directed graphs described below.\n\nGraph A:\n{metadata['graph1_description']}\n\n"
            f"Graph B:\n{metadata['graph2_description']}\n\n"
            "Do Graph A and Graph B have the exact same structure, just with different node labels? "
            "(In other words, are they isomorphic?)\n"
            "The answer is `True` or `False`."
        )

    def score_answer(self, answer, entry):
        return 1.0 if str(answer).strip().lower() == entry.answer.lower() else 0.0


@dataclass
class GraphSuccessorsConfig(Config):
    num_nodes: int = 6
    num_queries: int = 1
    max_hops: int = 2

    def update(self, c=1):
        self.num_nodes += c
        self.num_queries += c // 2
        self.max_hops += c


class GraphSuccessors(BaseGraphTask, Task):
    """DEPO-style k-th successor queries in a permutation digraph."""
    def __init__(self, config=GraphSuccessorsConfig()):
        super().__init__(config=config)

    def _jump(self, succ, x, k):
        for _ in range(k):
            x = succ[x]
        return x

    def generate(self):
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

        return Problem(
            metadata={
                "graph_description": self._render_graph(G),
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
            "Each pair (x, k) asks for the k-th successor of x (following exact directed edges k times).\n"
            "The answer is a Python list of integers in query order."
        )

    def score_answer(self, answer, entry):
        pred = _parse_list(answer)
        true = _parse_list(entry.answer)
        if pred is None or true is None or len(pred) != len(true):
            return 0.0
        return sum(p == t for p, t in zip(pred, true)) / len(true)


@dataclass
class GraphDependenciesConfig(Config):
    num_nodes: int = 6
    max_prereqs: int = 2

    def update(self, c=1):
        self.num_nodes += c
        self.max_prereqs += c // 2


class GraphDependencies(BaseGraphTask, Task):
    """BREVO-style recursive dependency resolution implemented via DAG topologies."""
    def __init__(self, config=GraphDependenciesConfig()):
        super().__init__(config=config)

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

    def generate(self):
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

            return Problem(
                metadata={
                    "graph_description": self._render_graph(G),
                    "query": q,
                    "nodes": list(G.nodes()),
                    "edges": list(G.edges()),
                },
                answer=str(answer),
            )
        return self.generate()

    def prompt(self, m):
        return (
            f"Consider the directed graph:\n\n{m['graph_description']}\n\n"
            f"In this scenario, a directed edge from U to V means V depends on U (so U is a prerequisite of V).\n"
            f"List all prerequisites of node {m['query']} (recursively), making sure to order base prerequisites first.\n"
            "Do not include the query node itself.\n"
            "If A is a prerequisite of B and both appear in your answer, A must appear before B.\n"
            "Tie-break nodes with no mutual dependency lexicographically (smaller node ID first).\n"
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
            # edge u->v means u is prereq of v, so u must come before v
            if pos[u] > pos[v]:  
                return 0.0
        return 1.0