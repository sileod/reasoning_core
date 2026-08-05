import re

import pytest

from reasoning_core import get_task, list_tasks, score_answer
from reasoning_core.tasks.logic_depth import (
    Atom,
    MultistepNLIConfig,
    PredSig,
    Rule,
    Theory,
    chase,
    render,
)
from reasoning_core.tasks.logic_derivation import LogicDerivation, canonical_trace


def unary_theory(facts, rules, domain_pack="surface"):
    preds = {atom.pred for atom in facts}
    preds |= {
        atom.pred
        for rule in rules
        for atom in (*rule.body, rule.head)
        if isinstance(atom, Atom)
    }
    return Theory(
        facts=facts,
        rules=rules,
        denials=[],
        pred_sigs={p: PredSig(p, ("person",)) for p in preds},
        entities={"person": ("alice",)},
        domain_pack=domain_pack,
    )


def one_step_trace(pack, fact, rule):
    theory = Theory(
        facts=[fact],
        rules=[rule],
        denials=[],
        pred_sigs={},
        entities={},
        domain_pack=pack,
    )
    res = chase(theory)
    target = rule.head
    trace = canonical_trace(theory, res, {fact: 0, rule: 1}, target)
    assert trace.unique
    return trace.answer


def test_canonical_trace_is_forward_and_uses_step_references():
    x = ("alice",)
    a, b, c, target = [Atom(p, x) for p in ("a", "b", "c", "target")]
    theory = unary_theory(
        [a, b],
        [Rule((a,), c), Rule((c, b), target)],
    )
    res = chase(theory)
    _, source, _ = render(theory)
    trace = canonical_trace(theory, res, source, target)

    assert trace.unique
    assert trace.depth == 2
    assert trace.size == 2
    assert trace.answer == (
        "2: 0 => alice is c\n"
        "3: @0 1 => alice is target"
    )


def test_canonical_trace_optimizes_proof_dag_steps_with_sharing():
    x = ("alice",)
    a, b, shared, left, right, u, v, target = [
        Atom(p, x)
        for p in ("a", "b", "shared", "left", "right", "u", "v", "target")
    ]
    theory = unary_theory(
        [a, b],
        [
            Rule((a,), shared),
            Rule((shared,), left),
            Rule((shared,), right),
            Rule((a,), u),
            Rule((u,), left),
            Rule((b,), v),
            Rule((v,), right),
            Rule((left, right), target),
        ],
    )
    res = chase(theory)
    _, source, _ = render(theory)
    trace = canonical_trace(theory, res, source, target)

    assert trace.unique
    assert trace.depth == 3
    assert trace.size == 4
    assert trace.answer == (
        "2: 0 => alice is shared\n"
        "3: @0 => alice is left\n"
        "4: @0 => alice is right\n"
        "9: @1 @2 => alice is target"
    )


def test_canonical_trace_rejects_equal_optimal_derivations():
    x = ("alice",)
    a, b, target = [Atom(p, x) for p in ("a", "b", "target")]
    theory = unary_theory(
        [a, b],
        [Rule((a,), target), Rule((b,), target)],
    )
    res = chase(theory)
    _, source, _ = render(theory)
    trace = canonical_trace(theory, res, source, target)

    assert not trace.unique
    assert trace.answer is None
    assert set(trace.traces) == {
        "2: 0 => alice is target",
        "3: 1 => alice is target",
    }


@pytest.mark.parametrize(
    ("pack", "fact", "rule", "expected"),
    [
        (
            "surface",
            Atom("trained", ("alice",)),
            Rule((Atom("trained", ("?x",)),), Atom("careful", ("?x",))),
            "1: 0 => alice is careful",
        ),
        (
            "abstract",
            Atom("p0", ("Alice",)),
            Rule((Atom("p0", ("?x",)),), Atom("p5", ("?x",))),
            "1: 0 => Alice is foxtrot tagged",
        ),
        (
            "kinship",
            Atom("adult", ("alice",)),
            Rule((Atom("adult", ("?x",)),), Atom("ancestor", ("?x", "bruno"))),
            "1: 0 => alice is an ancestor of bruno",
        ),
        (
            "spatial",
            Atom("marked", ("box",)),
            Rule((Atom("marked", ("?x",)),), Atom("left_of", ("?x", "key"))),
            "1: 0 => box is left of key",
        ),
    ],
)
def test_trace_conclusions_use_domain_surface_text(pack, fact, rule, expected):
    assert one_step_trace(pack, fact, rule) == expected


def test_negative_conclusions_use_surface_negation():
    fact = Atom("trained", ("alice",))
    rule = Rule((Atom("trained", ("?x",)),), Atom("flagged", ("?x",), False))
    assert one_step_trace("surface", fact, rule) == "1: 0 => alice is not flagged"

    fact = Atom("p0", ("Alice",))
    rule = Rule((Atom("p0", ("?x",)),), Atom("p5", ("?x",), False))
    assert one_step_trace("abstract", fact, rule) == "1: 0 => Alice is not foxtrot tagged"


@pytest.mark.parametrize("pack", ("surface", "abstract", "kinship", "spatial"))
def test_each_domain_pack_generates_and_scores(pack):
    cfg = MultistepNLIConfig(domain_packs=(pack,))
    task = LogicDerivation(config=cfg)
    ex = task.generate_example(max_tokens=0)

    assert ex.metadata.domain_pack == pack
    assert ex.answer
    assert score_answer(ex.answer, ex) == 1
    assert all(
        int(ref[1:]) < i
        for i, line in enumerate(ex.answer.splitlines())
        for ref in line.split()
        if ref.startswith("@")
    )
    if pack == "abstract":
        assert not re.search(r"\b[pr]\d+\(", ex.answer)
        assert " tagged" in ex.answer or "-linked" in ex.answer or "related" in ex.answer or "connected" in ex.answer or "associated" in ex.answer


def test_logic_derivation_registers_generates_and_scores():
    assert "logic_derivation" in list_tasks()
    task = get_task("logic_derivation")
    ex = task.generate_example(max_tokens=0)

    assert ex.answer
    assert "=>" in ex.answer
    assert ex.metadata.target
    assert ex.metadata.proof_depth >= 2
    assert ex.metadata.proof_steps >= ex.metadata.proof_depth
    assert ex.metadata.optimal_trace_count == 1
    assert score_answer(ex.answer, ex) == 1

    spaced = ex.answer.replace(":", " : ").replace("=>", " => ")
    assert task.score_answer(spaced, ex) == 1
    assert task.score_answer(ex.answer + "\n0: 0 => alice is bogus", ex) == 0
