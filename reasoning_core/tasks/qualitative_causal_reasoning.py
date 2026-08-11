from dataclasses import dataclass
import random

import networkx as nx
import numpy as np
from networkx.algorithms.d_separation import is_d_separator

from reasoning_core.template import Config, Entry, Task


SIGNS = {"+": 1, "-": -1}
ANSWER_SPACES = {
    "intervention": ["increase", "decrease", "no_effect", "ambiguous"],
    "marginal_association": [
        "increase",
        "decrease",
        "no_association",
        "ambiguous",
    ],
    "conditional_association": ["associated", "independent"],
}
LABELS = sorted({label for labels in ANSWER_SPACES.values() for label in labels})
SEMANTIC_ASSUMPTION = (
    "Assume linear causal relations, independent noise, and no exact cancellations."
)


@dataclass(frozen=True)
class Query:
    kind: str
    source: str
    target: str
    conditioned: frozenset = frozenset()


@dataclass
class Instance:
    graph: nx.DiGraph
    query: Query
    answer: str
    phenomenon: str
    n_extra: int
    p_edge: float


@dataclass
class QualitativeCausalReasoningConfig(Config):
    n_extra: int = 8
    p_edge: float = 0.10

    def apply_difficulty(self, level):
        self.n_extra += 3 * level
        self.p_edge = min(0.20, self.p_edge + 0.015 * level)


def combine_signs(signs):
    signs = set(signs)
    if not signs:
        return "no_effect"
    if signs == {1}:
        return "increase"
    if signs == {-1}:
        return "decrease"
    return "ambiguous"


def edge_sign(G, u, v):
    return G.edges[u, v]["sign"]


def add_edge(G, u, v, sign):
    G.add_edge(u, v, sign=SIGNS[sign])


def intervention_graph(G, x):
    H = G.copy()
    H.remove_edges_from(list(H.in_edges(x)))
    return H


def active_signed_effects(G, source, target, conditioned=frozenset()):
    if source in conditioned or target in conditioned:
        return set()

    H = G.copy()
    for z in conditioned:
        H.remove_edges_from(list(H.in_edges(z)))
        H.remove_edges_from(list(H.out_edges(z)))

    return directed_path_signs(H, source)[target]


def directed_path_signs(G, source):
    """Signs of all directed paths from source to every node in a DAG."""
    signs = {u: set() for u in G.nodes}
    signs[source].add(1)
    for u in nx.topological_sort(G):
        for v in G.successors(u):
            for sign in signs[u]:
                signs[v].add(sign * edge_sign(G, u, v))
    return signs


def verify_intervention(G, query):
    H = intervention_graph(G, query.source)
    return combine_signs(
        active_signed_effects(H, query.source, query.target, query.conditioned)
    )


def is_collider(G, a, b, c):
    return G.has_edge(a, b) and G.has_edge(c, b)


def undirected_edge_sign(G, u, v):
    if G.has_edge(u, v):
        return edge_sign(G, u, v)
    return edge_sign(G, v, u)


def verify_marginal_association(G, query):
    if query.conditioned:
        raise ValueError("marginal association queries cannot condition on nodes")
    signs = set()
    neighbors = {node: tuple(G.predecessors(node)) + tuple(G.successors(node))
                 for node in G.nodes}
    visited = {query.source}

    def search(previous, current, sign):
        for nxt in neighbors[current]:
            if nxt in visited or (
                previous is not None and is_collider(G, previous, current, nxt)
            ):
                continue
            next_sign = sign * undirected_edge_sign(G, current, nxt)
            if nxt == query.target:
                signs.add(next_sign)
                if len(signs) == 2:
                    return True
                continue
            visited.add(nxt)
            if search(current, nxt, next_sign):
                return True
            visited.remove(nxt)
        return False

    search(None, query.source, 1)

    if not signs:
        return "no_association"
    if signs == {1}:
        return "increase"
    if signs == {-1}:
        return "decrease"
    return "ambiguous"


def verify_conditional_association(G, query):
    separated = is_d_separator(
        G,
        {query.source},
        {query.target},
        set(query.conditioned),
    )
    return "independent" if separated else "associated"


def verify(G, query):
    if query.kind == "intervention":
        return verify_intervention(G, query)
    if query.kind == "marginal_association":
        return verify_marginal_association(G, query)
    if query.kind == "conditional_association":
        return verify_conditional_association(G, query)
    raise ValueError(query.kind)


def kernel_direct(label, rng):
    G = nx.DiGraph()
    sign = "+" if label == "increase" else "-"
    add_edge(G, "X", "Y", sign)
    return G, Query("intervention", "X", "Y")


def kernel_mediator(label, rng):
    G = nx.DiGraph()
    s1 = rng.choice(["+", "-"])
    s2 = "+" if (label == "increase") == (s1 == "+") else "-"
    add_edge(G, "X", "M", s1)
    add_edge(G, "M", "Y", s2)
    return G, Query("intervention", "X", "Y")


def kernel_blocked_mediator(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "M", "+")
    add_edge(G, "M", "Y", "+")
    return G, Query("intervention", "X", "Y", frozenset({"M"}))


def kernel_competing_paths(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "A", "+")
    add_edge(G, "A", "Y", "+")
    add_edge(G, "X", "B", "+")
    add_edge(G, "B", "Y", "-")
    return G, Query("intervention", "X", "Y")


def kernel_common_cause(label, rng):
    G = nx.DiGraph()
    s1 = rng.choice(["+", "-"])
    s2 = "+" if (label == "increase") == (s1 == "+") else "-"
    add_edge(G, "Z", "X", s1)
    add_edge(G, "Z", "Y", s2)
    return G, Query("marginal_association", "X", "Y")


def kernel_closed_collider(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "C", "+")
    add_edge(G, "Y", "C", "+")
    return G, Query("conditional_association", "X", "Y")


def kernel_open_collider(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "C", "+")
    add_edge(G, "Y", "C", "+")
    return G, Query("conditional_association", "X", "Y", frozenset({"C"}))


def kernel_descendant_open_collider(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "C", "+")
    add_edge(G, "Y", "C", "+")
    add_edge(G, "C", "D", "+")
    return G, Query("conditional_association", "X", "Y", frozenset({"D"}))


def kernel_conditioned_chain(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "M", "+")
    add_edge(G, "M", "Y", "+")
    return G, Query("conditional_association", "X", "Y", frozenset({"M"}))


def kernel_conditioned_common_cause(rng):
    G = nx.DiGraph()
    add_edge(G, "Z", "X", "+")
    add_edge(G, "Z", "Y", "+")
    return G, Query("conditional_association", "X", "Y", frozenset({"Z"}))


def kernel_multiple_treks(competing, rng):
    G = nx.DiGraph()
    add_edge(G, "A", "X", "+")
    add_edge(G, "A", "Y", "+")
    add_edge(G, "B", "X", "+")
    add_edge(G, "B", "Y", "-" if competing else "+")
    return G, Query("marginal_association", "X", "Y")


def kernel_disconnected_marginal(rng):
    G = nx.DiGraph()
    G.add_nodes_from(["X", "Y"])
    return G, Query("marginal_association", "X", "Y")


def kernel_blocked_collider_marginal(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "C", "+")
    add_edge(G, "Y", "C", "+")
    return G, Query("marginal_association", "X", "Y")


def kernel_observe_vs_do_association(rng):
    G = nx.DiGraph()
    add_edge(G, "Z", "X", "+")
    add_edge(G, "Z", "Y", "+")
    return G, Query("marginal_association", "X", "Y")


def kernel_observe_vs_do_intervention(rng):
    G = nx.DiGraph()
    add_edge(G, "Z", "X", "+")
    add_edge(G, "Z", "Y", "+")
    return G, Query("intervention", "X", "Y")


def kernel_reverse_path(rng):
    G = nx.DiGraph()
    add_edge(G, "Y", "M", "+")
    add_edge(G, "M", "X", "+")
    return G, Query("intervention", "X", "Y")


def kernel_blocked_negative_mediator(rng):
    G = nx.DiGraph()
    add_edge(G, "X", "M", "-")
    add_edge(G, "M", "Y", "+")
    return G, Query("intervention", "X", "Y", frozenset({"M"}))


KERNELS = {
    "direct_intervention": (["increase", "decrease"], kernel_direct),
    "mediated_intervention": (["increase", "decrease"], kernel_mediator),
    "blocked_mediator": (["no_effect"], lambda label, rng: kernel_blocked_mediator(rng)),
    "competing_paths": (["ambiguous"], lambda label, rng: kernel_competing_paths(rng)),
    "common_cause_association": (["increase", "decrease"], kernel_common_cause),
    "closed_collider": (["independent"], lambda label, rng: kernel_closed_collider(rng)),
    "open_collider_explaining_away": (["associated"], lambda label, rng: kernel_open_collider(rng)),
    "descendant_of_collider": (["associated"], lambda label, rng: kernel_descendant_open_collider(rng)),
    "conditioned_chain": (["independent"], lambda label, rng: kernel_conditioned_chain(rng)),
    "conditioned_common_cause": (["independent"], lambda label, rng: kernel_conditioned_common_cause(rng)),
    "multiple_same_sign_treks": (["increase"], lambda label, rng: kernel_multiple_treks(False, rng)),
    "competing_sign_treks": (["ambiguous"], lambda label, rng: kernel_multiple_treks(True, rng)),
    "disconnected_marginal_pair": (["no_association"], lambda label, rng: kernel_disconnected_marginal(rng)),
    "blocked_collider_marginal": (["no_association"], lambda label, rng: kernel_blocked_collider_marginal(rng)),
    "confounded_observation": (["increase"], lambda label, rng: kernel_observe_vs_do_association(rng)),
    "confounded_intervention": (["no_effect"], lambda label, rng: kernel_observe_vs_do_intervention(rng)),
    "reverse_path_no_effect": (["no_effect"], lambda label, rng: kernel_reverse_path(rng)),
    "blocked_negative_mediator": (["no_effect"], lambda label, rng: kernel_blocked_negative_mediator(rng)),
}


def _augmented_topological_order(G, extra, rng):
    order = list(nx.topological_sort(G))
    for node in rng.permutation(extra):
        order.insert(int(rng.integers(len(order) + 1)), node)
    return order


def augment(G, rng, n_extra=8, p_edge=0.12):
    H = G.copy()
    extra = [f"N{i}" for i in range(n_extra)]
    H.add_nodes_from(extra)

    order = _augmented_topological_order(G, extra, rng)
    for i, u in enumerate(order):
        for v in order[i + 1 :]:
            if not H.has_edge(u, v) and rng.random() < p_edge:
                H.add_edge(u, v, sign=int(rng.choice([1, -1])))

    if not nx.is_directed_acyclic_graph(H):
        raise RuntimeError("augment produced cycle")
    return H


def relabel_graph_and_query(G, q, rng):
    names = list(G.nodes)
    shuffled = list(rng.permutation([f"X{i}" for i in range(len(names))]))
    mapping = dict(zip(names, shuffled))
    H = nx.relabel_nodes(G, mapping, copy=True)
    query = Query(
        q.kind,
        mapping[q.source],
        mapping[q.target],
        frozenset(mapping[z] for z in q.conditioned),
    )
    return H, query


def quality_ok(G, q, answer, phenomenon=None):
    if q.source == q.target:
        return False
    connected = nx.has_path(G.to_undirected(), q.source, q.target)
    if q.kind == "intervention" and answer == "no_effect":
        H = intervention_graph(G, q.source)
        for node in q.conditioned:
            H.remove_edges_from(list(H.in_edges(node)))
            H.remove_edges_from(list(H.out_edges(node)))
        return connected and not nx.has_path(H, q.source, q.target)
    if q.kind == "marginal_association" and answer == "no_association":
        return connected or phenomenon == "disconnected_marginal_pair"
    if q.kind == "conditional_association" and answer == "independent":
        return connected
    return True


def sample_instance(seed=None, n_extra=8, p_edge=0.10):
    rng = np.random.default_rng(seed)
    query_kind = str(rng.choice(list(ANSWER_SPACES)))
    target_label = str(rng.choice(ANSWER_SPACES[query_kind]))
    eligible = [
        (phenomenon, make)
        for phenomenon, (labels, make) in KERNELS.items()
        if target_label in labels
    ]

    for _ in range(1000):
        phenomenon, make = eligible[int(rng.integers(len(eligible)))]

        G, q = make(target_label, rng)
        if q.kind != query_kind:
            continue
        before = verify(G, q)

        H = augment(G, rng, n_extra=n_extra, p_edge=p_edge)
        ans = verify(H, q)
        if ans != before:
            continue

        H, relabeled_q = relabel_graph_and_query(H, q, rng)
        ans = verify(H, relabeled_q)
        if ans == target_label and quality_ok(H, relabeled_q, ans, phenomenon):
            return Instance(H, relabeled_q, ans, phenomenon, n_extra, p_edge)

    raise RuntimeError("failed to sample valid qualitative causal instance")


def render_edge_list(G):
    return "\n".join(
        f"- {u} directly {'increases' if d['sign'] == 1 else 'decreases'} {v}."
        for u, v, d in sorted(G.edges(data=True))
    )


def render_compact(G):
    edges = [
        f"{u} directly {'increases' if d['sign'] == 1 else 'decreases'} {v}"
        for u, v, d in sorted(G.edges(data=True))
    ]
    return "; ".join(edges) + "."


def render_table(G):
    return render_edge_list(G)


RENDERERS = [render_edge_list, render_compact, render_table]


class QualitativeCausalReasoning(Task):
    summary = "Reason qualitatively about causal effects and associations in graphs."

    def __init__(self, config=None):
        super().__init__(config=config or QualitativeCausalReasoningConfig())

    def generate_entry(self):
        instance = sample_instance(
            n_extra=self.config.n_extra,
            p_edge=self.config.p_edge,
        )
        edges = [
            (u, v, "+" if d["sign"] == 1 else "-")
            for u, v, d in sorted(instance.graph.edges(data=True))
        ]
        metadata = {
            "edges": edges,
            "nodes": sorted(instance.graph.nodes),
            "query": {
                "kind": instance.query.kind,
                "source": instance.query.source,
                "target": instance.query.target,
                "conditioned": sorted(instance.query.conditioned),
            },
            "query_kind": instance.query.kind,
            "phenomenon": instance.phenomenon,
            "n_extra": instance.n_extra,
            "p_edge": instance.p_edge,
            "semantics": "linear_sem_independent_noise_no_cancellation",
            "render_style": random.choice(["edge_list", "compact", "table"]),
        }
        return Entry(metadata=metadata, answer=instance.answer)

    def render_prompt(self, metadata):
        G = nx.DiGraph()
        for u, v, sign in metadata.edges:
            G.add_edge(u, v, sign=SIGNS[sign])
        qd = metadata.query
        q = Query(qd["kind"], qd["source"], qd["target"], frozenset(qd["conditioned"]))
        renderers = {
            "edge_list": render_edge_list,
            "compact": render_compact,
            "table": render_table,
        }
        graph_text = renderers[metadata.render_style](G)

        if q.kind == "intervention":
            cond = ""
            if q.conditioned:
                cond = " while holding " + ", ".join(sorted(q.conditioned)) + " fixed"
            query_text = f"If we intervene to increase {q.source}{cond}, what happens to {q.target}?"
        elif q.kind == "marginal_association":
            query_text = (
                f"Without conditioning, how is {q.source} associated with {q.target}?"
            )
        elif q.kind == "conditional_association":
            if q.conditioned:
                given = ", ".join(sorted(q.conditioned))
                query_text = f"Given {given}, are {q.source} and {q.target} associated?"
            else:
                query_text = (
                    f"Without conditioning, are {q.source} and {q.target} associated?"
                )
        else:
            raise ValueError(q.kind)

        labels = ANSWER_SPACES[q.kind]
        choices = (
            f"{labels[0]} or {labels[1]}"
            if len(labels) == 2
            else ", ".join(labels[:-1]) + f", or {labels[-1]}"
        )
        return (
            SEMANTIC_ASSUMPTION
            + "\n\n"
            + graph_text
            + f"\n\n{query_text}"
            + "\nAnswer with: "
            + choices
            + "."
        )

    def score_answer(self, answer, entry):
        normalized = str(answer).strip().lower()
        return 1 if normalized == entry.answer else 0

    def balancing_key(self, problem):
        return f"{problem.metadata.query_kind}:{problem.answer}"
