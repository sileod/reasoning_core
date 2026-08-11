import networkx as nx

from reasoning_core import get_task, list_tasks
from reasoning_core.tasks.math_lean import (
    BANNED_LEAN_TOKENS,
    LeanConfig,
    gen_forward_order_graph,
    get_runner,
)


def test_current_lean_tasks_are_discoverable():
    assert {"lean_missing_line", "lean_candidate_compilation"} <= set(list_tasks())
    assert type(get_task("LeanMissingLine", use_mathlib=False)).__name__ == "LeanMissingLine"


def test_forward_graph_core_profile_verifies_and_is_well_formed():
    inst = gen_forward_order_graph(LeanConfig(use_mathlib=False))

    assert inst is not None
    assert nx.is_directed_acyclic_graph(inst.G)
    assert inst.stats.final_verifies
    assert inst.stats.proof_depth >= 3
    assert inst.stats.useful_premises >= 2
    assert not any(tok in inst.theorem.lower() for tok in BANNED_LEAN_TOKENS)
    assert get_runner(use_mathlib=False).check(inst.theorem)[0]

    seen = set(inst.leaf_hyp_names.values())
    for line in inst.proof.splitlines():
        if line.startswith("have "):
            name = line.split()[1]
            refs = {
                token for token in line.replace(":", " ").split()
                if token.startswith("h") and token[1:].isdigit()
            }
            assert refs - {name} <= seen
            seen.add(name)

    for node in inst.G.nodes:
        data = inst.G.nodes[node]["data"]
        assert data.clause_formula
        assert data.full_cnf_clause
        if inst.G.in_degree(node):
            assert data.inference
            assert set(data.parents) == set(inst.G.predecessors(node))
