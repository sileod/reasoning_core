import networkx as nx
import random
from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass
from ast import literal_eval

# --- Configuration for All Graph Tasks ---
@dataclass
class GraphReasoningConfig(Config):
    num_nodes: int = 5  #needs 5 to avoid duplicates
    def update(self, c): 
        self.num_nodes *= (1+c)


_GRAPH_GENERATORS = [
    (nx.fast_gnp_random_graph, {'p': (0.15, 0.4)}),
    (nx.watts_strogatz_graph, {'k': (2, 4), 'p': (0.1, 0.3)}),
    (nx.barabasi_albert_graph, {'m': (1, 3)}),
    (nx.random_regular_graph, {'d': (2, 4)}), # Every node has d neighbors
    (nx.lollipop_graph, {'m': (4, 8), 'n': (2, 4)}), # A clique connected to a path
    (nx.grid_2d_graph, {'m': (3, 5), 'n': (3, 5)}), # A grid
]


class BaseGraphTask:
    """Handles shared, flexible graph generation and rendering."""
    def __init__(self, config=GraphReasoningConfig()):
        super().__init__(config)

    def _generate_graph(self):
        """Randomly selects a topology from the list and generates a graph."""
        num_nodes = self.config.num_nodes
        
        for _ in range(10): # Try a few times to get a valid graph
            gen_func, params_ranges = random.choice(_GRAPH_GENERATORS)
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
                    return G
            except (nx.NetworkXError, ValueError) as e:
                continue # Some generators can fail with certain params, just retry
        
        return nx.fast_gnp_random_graph(num_nodes, 0.5) # Fallback

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
    def generate(self):
        G = self._generate_graph()
        start, end = random.sample(list(G.nodes()), 2)
        path = nx.shortest_path(G, source=start, target=end)

        metadata = {
            "graph_description": self._render_graph(G), "start_node": start, "end_node": end,
            "nodes": list(G.nodes()), "edges": list(G.edges()), "optimal_length": len(path),
        }
        return Problem(metadata=metadata, answer=str(path))

    def prompt(self, m):
        return (f"Consider the graph:\n\n{m['graph_description']}\n\n"
                f"Find the shortest path from Node {m['start_node']} to Node {m['end_node']}.\n"
                "Answer with a Python list of integers. Example: `[0, 5, 3, 9]`.")

def score_answer(self, answer, entry):
        try: pred_path = literal_eval(answer)
        except: return 0.0
        if not isinstance(pred_path, list) or len(pred_path) < 1: return 0.0
        
        meta = entry.metadata
        
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

        if (pred_path[0] != start_node or pred_path[-1] != end_node or not nx.is_path(G, pred_path)):
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
            "Your answer must be a Python list of node integers, sorted in increasing order. "
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

class GraphCycleDetection(BaseGraphTask, Task):
    """Task to identify the specific nodes that form a cycle in a graph."""

    def generate(self):
        # Create a graph with exactly one cycle.
        # Start with a path graph (guaranteed acyclic), then add one edge.
        G = nx.path_graph(self.config.num_nodes)
        
        # Add one edge between non-adjacent nodes to create a single cycle.
        possible_edges = list(nx.non_edges(G))
        if not possible_edges: # Should not happen for n > 2
            return self.generate() # Retry
        u, v = random.choice(possible_edges)
        G.add_edge(u, v)

        # The answer is the set of nodes forming this unique cycle.
        cycle_edges = nx.find_cycle(G)
        answer_nodes = sorted(list(set(node for edge in cycle_edges for node in edge)))
        
        metadata = {"graph_description": self._render_graph(G)}
        return Problem(metadata=metadata, answer=str(answer_nodes))

    def prompt(self, metadata):
        return (
            f"Consider the graph below, which contains exactly one cycle.\n\n"
            f"{metadata['graph_description']}\n\n"
            "Identify all the nodes that form the cycle.\n"
            "Your answer must be a Python list of node integers, sorted in increasing order. "
            "Example: `[2, 5, 7, 8]`."
        )

    def score_answer(self, answer, entry):
        """Scores based on whether the predicted set of nodes matches the true cycle."""
        try:
            pred_nodes = literal_eval(answer)
            true_nodes = literal_eval(entry.answer)
            # Use sets for order-agnostic comparison, then check if sorted.
            is_correct_set = (set(pred_nodes) == set(true_nodes))
            is_sorted = (pred_nodes == sorted(pred_nodes))
            return 1.0 if is_correct_set and is_sorted else 0.0
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
            "Answer with only `True` or `False`."
        )

    def score_answer(self, answer, entry):
        return 1.0 if str(answer).strip().lower() == entry.answer.lower() else 0.0