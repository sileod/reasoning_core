import os

import networkx as nx
import pytest

from reasoning_core.template import Problem
from reasoning_core.tasks._tptp_sat_graph import DerivationNode
from reasoning_core.tasks.deprecated.math_tptp_dev import ResolutionStep, ResolutionStepConfig
from reasoning_core.tasks.math_tptp import (
    TPTPTerm,
    canonical,
    parse_clause,
    prove_conjecture,
    render_clause,
    resolvents,
    unify,
)


def test_canonical_round_trip_and_variable_ordering():
    clauses = [
        "(~q(X9) | p(X9,f(X2)))",
        "(X7 = f(X3) | X3 != a)",
        "$false",
    ]
    for text in clauses:
        clause = parse_clause(text)
        assert canonical(parse_clause(render_clause(clause))) == canonical(clause)

    assert canonical(parse_clause(clauses[0])) == "(p(X1,f(X2)) | ~q(X1))"


def test_canonical_rejects_masked_order_ties():
    assert canonical(parse_clause("(p(X1) | p(X2))")) is None


def test_equality_and_occurs_check():
    positive, negative = parse_clause("(f(X1) = a | f(X2) != a)")
    assert positive.pred == negative.pred == "="
    assert positive.sign is True
    assert negative.sign is False

    variable = TPTPTerm("X1")
    recursive = TPTPTerm("f", (variable,))
    assert unify(variable, recursive) is None


def test_worked_set_theory_resolvent_is_canonical():
    expected = "(member(X1,X2) | ~member(X1,intersection(X2,X3)))"
    assert canonical(parse_clause(expected)) == expected


def test_unique_binary_resolvent():
    clause_a = parse_clause("(member(X1,X2) | ~bridge(X1,X2,X3))")
    clause_b = parse_clause("(bridge(Y1,Y2,Y3) | disjoint(Y2,Y3))")
    possible = resolvents(clause_a, clause_b)
    assert len(possible) == 1
    assert canonical(possible[0]) == "(disjoint(X1,X2) | member(X3,X1))"


def test_mining_validates_child_and_renames_parents_apart():
    graph = nx.DiGraph()
    graph.add_node(
        "a",
        data=DerivationNode(
            "a",
            "(member(X1,X2) | ~bridge(X1,X2,X3))",
        ),
    )
    graph.add_node(
        "b",
        data=DerivationNode(
            "b",
            "(bridge(X1,X2,X3) | disjoint(X2,X3))",
        ),
    )
    graph.add_node(
        "c",
        data=DerivationNode(
            "c",
            "(disjoint(X2,X3) | member(X1,X2))",
            inference="inference(resolution,[status(thm)],[a,b])",
        ),
    )
    graph.add_edges_from([("a", "c"), ("b", "c")])

    task = ResolutionStep.__new__(ResolutionStep)
    task.config = ResolutionStepConfig(min_total_literals=4, min_term_depth=0)
    task.graph = graph
    task.axiom_set = "SET001-0.ax"
    task.pool = []
    task._mine_pool()

    assert len(task.pool) == 1
    problem = task.pool[0]
    assert problem.answer == "(disjoint(X1,X2) | member(X3,X1))"
    assert "Y1" in problem.metadata.clause_b
    assert problem.metadata.rule == "resolution"
    assert problem.metadata.total_literals == 4


def test_score_answer_canonicalizes_and_has_string_fallback():
    task = ResolutionStep.__new__(ResolutionStep)
    entry = Problem({}, "(p(X1) | ~q(X1))")

    assert task.score_answer("(~q(Z8)|p(Z8))", entry) == 1.0
    assert task.score_answer("`(p(X1) | ~q(X1))`", entry) == 1.0
    assert task.score_answer("(p(X1) | q(X1))", entry) == 0.0
    assert task.score_answer("(p(X1) | ~q(X1)", entry) == 0.0


@pytest.mark.skipif(
    not os.environ.get("RUN_TPTP_PROVER_TESTS"),
    reason="requires the external prover runtime",
)
def test_resolvent_is_entailed_by_parents():
    parents = [
        "(member(X1,X2) | ~bridge(X1,X2,X3))",
        "(bridge(Y1,Y2,Y3) | disjoint(Y2,Y3))",
    ]
    result = canonical(resolvents(*(parse_clause(parent) for parent in parents))[0])
    assert prove_conjecture(parents, result, time_limit_seconds="10") is True
