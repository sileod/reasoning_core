import json
import random

import networkx as nx
import numpy as np

from reasoning_core.tasks.qualitative_causal_reasoning import (
    ANSWER_SPACES,
    KERNELS,
    QualitativeCausalReasoning,
    Query,
    add_edge,
    kernel_blocked_mediator,
    kernel_competing_paths,
    sample_instance,
    verify,
    verify_conditional_association,
    verify_intervention,
    verify_marginal_association,
)
from reasoning_core.template import edict


def graph(*edges):
    G = nx.DiGraph()
    for u, v, sign in edges:
        add_edge(G, u, v, sign)
    return G


def test_direct_intervention_signs():
    positive = graph(("X", "Y", "+"))
    negative = graph(("X", "Y", "-"))
    query = Query("intervention", "X", "Y")

    assert verify_intervention(positive, query) == "increase"
    assert verify_intervention(negative, query) == "decrease"


def test_positive_and_negative_directed_paths_are_ambiguous():
    G, query = kernel_competing_paths(np.random.default_rng(0))
    assert verify_intervention(G, query) == "ambiguous"


def test_held_fixed_mediator_blocks_intervention_effect():
    G, query = kernel_blocked_mediator(np.random.default_rng(0))
    assert verify_intervention(G, query) == "no_effect"


def test_common_cause_marginal_and_intervention_contrast():
    same = graph(("Z", "X", "+"), ("Z", "Y", "+"))
    opposite = graph(("Z", "X", "+"), ("Z", "Y", "-"))

    marginal = Query("marginal_association", "X", "Y")
    intervention = Query("intervention", "X", "Y")
    assert verify_marginal_association(same, marginal) == "increase"
    assert verify_intervention(same, intervention) == "no_effect"
    assert verify_marginal_association(opposite, marginal) == "decrease"


def test_collider_is_marginally_blocked_and_opened_by_conditioning():
    G = graph(("X", "C", "+"), ("Y", "C", "+"))

    marginal = Query("marginal_association", "X", "Y")
    given_c = Query(
        "conditional_association", "X", "Y", frozenset({"C"})
    )
    assert verify_marginal_association(G, marginal) == "no_association"
    assert verify_conditional_association(G, given_c) == "associated"


def test_descendant_of_collider_opens_path():
    G = graph(
        ("X", "C", "+"),
        ("Y", "C", "+"),
        ("C", "D", "+"),
    )
    query = Query(
        "conditional_association", "X", "Y", frozenset({"D"})
    )
    assert verify_conditional_association(G, query) == "associated"


def test_conditioning_blocks_chain_and_fork():
    chain = graph(("X", "M", "+"), ("M", "Y", "+"))
    fork = graph(("Z", "X", "+"), ("Z", "Y", "+"))

    assert verify_conditional_association(
        chain,
        Query("conditional_association", "X", "Y", frozenset({"M"})),
    ) == "independent"
    assert verify_conditional_association(
        fork,
        Query("conditional_association", "X", "Y", frozenset({"Z"})),
    ) == "independent"


def test_competing_sign_treks_are_ambiguous():
    G = graph(
        ("A", "X", "+"),
        ("A", "Y", "+"),
        ("B", "X", "+"),
        ("B", "Y", "-"),
    )
    query = Query("marginal_association", "X", "Y")
    assert verify_marginal_association(G, query) == "ambiguous"


def test_multiple_same_sign_treks_have_that_sign():
    G = graph(
        ("A", "X", "+"),
        ("A", "Y", "+"),
        ("B", "X", "-"),
        ("B", "Y", "-"),
    )
    query = Query("marginal_association", "X", "Y")
    assert verify_marginal_association(G, query) == "increase"


def test_marginal_dynamic_program_matches_path_enumeration():
    def brute_force(G, query):
        signs = set()
        for path in nx.all_simple_paths(G.to_undirected(), query.source, query.target):
            if any(
                G.has_edge(a, b) and G.has_edge(c, b)
                for a, b, c in zip(path, path[1:], path[2:])
            ):
                continue
            sign = 1
            for left, right in zip(path, path[1:]):
                edge = (left, right) if G.has_edge(left, right) else (right, left)
                sign *= G.edges[edge]["sign"]
            signs.add(sign)
        if not signs:
            return "no_association"
        if signs == {1}:
            return "increase"
        if signs == {-1}:
            return "decrease"
        return "ambiguous"

    rng = random.Random(0)
    for _ in range(100):
        G = nx.DiGraph()
        G.add_nodes_from(range(7))
        for left in range(7):
            for right in range(left + 1, 7):
                if rng.random() < 0.35:
                    G.add_edge(left, right, sign=rng.choice((-1, 1)))
        for source in range(7):
            for target in range(source + 1, 7):
                query = Query("marginal_association", source, target)
                assert verify_marginal_association(G, query) == brute_force(G, query)


def test_sample_instance_is_deterministic_and_verified():
    first = sample_instance(seed=123)
    second = sample_instance(seed=123)

    assert first.answer == second.answer
    assert sorted(first.graph.edges(data="sign")) == sorted(
        second.graph.edges(data="sign")
    )
    assert first.query == second.query
    assert verify(first.graph, first.query) == first.answer


def test_task_uses_query_specific_prompts_and_new_semantics_metadata():
    task = QualitativeCausalReasoning()

    for seed in range(30):
        instance = sample_instance(seed=seed)
        edges = [
            (u, v, "+" if data["sign"] == 1 else "-")
            for u, v, data in sorted(instance.graph.edges(data=True))
        ]
        metadata = edict({
            "edges": edges,
            "query": {
                "kind": instance.query.kind,
                "source": instance.query.source,
                "target": instance.query.target,
                "conditioned": sorted(instance.query.conditioned),
            },
            "render_style": "edge_list",
        })
        prompt = task.render_prompt(metadata)

        assert "Assume linear causal relations, independent noise" in prompt
        assert " tends to " not in prompt
        assert "collider" not in prompt.lower()
        assert instance.answer in ANSWER_SPACES[instance.query.kind]

    problem = task.generate_example(max_tokens=0)
    serialized = json.dumps(dict(problem.metadata)).lower()
    assert problem.metadata.semantics == (
        "linear_sem_independent_noise_no_cancellation"
    )
    assert "association_convention" not in problem.metadata
    assert "collider" not in problem.prompt.lower()
    assert "flip" not in serialized and "inversion" not in serialized
    assert task.score_answer(problem.answer, problem) == 1
    assert task.score_answer("not_a_label", problem) == 0


def test_each_kernel_declares_its_exact_reachable_labels():
    rng = np.random.default_rng(0)

    for labels, make in KERNELS.values():
        for label in labels:
            G, query = make(label, rng)
            assert label in ANSWER_SPACES[query.kind]
            assert verify(G, query) == label
