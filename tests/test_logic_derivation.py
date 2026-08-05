from reasoning_core import get_task, list_tasks, score_answer
from reasoning_core.tasks.logic_depth import Atom, PredSig, Rule, Theory, chase, render
from reasoning_core.tasks.logic_derivation import canonical_proof


def unary_theory(facts, rules):
    preds = {atom.pred for atom in facts}
    preds |= {atom.pred for rule in rules for atom in (*rule.body, rule.head)}
    return Theory(
        facts=facts,
        rules=rules,
        denials=[],
        pred_sigs={p: PredSig(p, ("person",)) for p in preds},
        entities={"person": ("alice",)},
    )


def test_canonical_proof_prefers_fewer_nodes_at_equal_depth():
    x = ("alice",)
    a, b, c, d, target = [Atom(p, x) for p in ("a", "b", "c", "d", "target")]
    theory = unary_theory(
        [a, b],
        [
            Rule((a,), c),
            Rule((c,), target),
            Rule((a, b), d),
            Rule((d,), target),
        ],
    )
    res = chase(theory)
    _, source, _ = render(theory)
    proof = canonical_proof(theory, res, source, target)

    assert proof.unique
    assert proof.depth == 2
    assert proof.size == 3
    assert proof.answer == "3(2(0))"


def test_canonical_proof_rejects_equal_optimal_derivations():
    x = ("alice",)
    a, b, target = [Atom(p, x) for p in ("a", "b", "target")]
    theory = unary_theory(
        [a, b],
        [Rule((a,), target), Rule((b,), target)],
    )
    res = chase(theory)
    _, source, _ = render(theory)
    proof = canonical_proof(theory, res, source, target)

    assert not proof.unique
    assert proof.answer is None
    assert {p.render() for p in proof.proofs} == {"2(0)", "3(1)"}


def test_logic_derivation_registers_generates_and_scores():
    assert "logic_derivation" in list_tasks()
    task = get_task("logic_derivation")
    ex = task.generate_example(max_tokens=0)

    assert ex.answer
    assert all(c.isdigit() or c in "()," for c in ex.answer)
    assert ex.metadata.target
    assert ex.metadata.proof_depth >= 2
    assert ex.metadata.proof_size >= ex.metadata.proof_depth + 1
    assert ex.metadata.optimal_proof_count == 1
    assert score_answer(ex.answer, ex) == 1

    spaced = ex.answer.replace("(", " ( ").replace(",", " , ").replace(")", " ) ")
    assert task.score_answer(spaced, ex) == 1
    assert task.score_answer(ex.answer + "0", ex) == 0
