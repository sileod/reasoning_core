from reasoning_core import get_task, list_tasks, score_answer
from reasoning_core.tasks.logic_depth import Atom, PredSig, Rule, Theory, chase, render
from reasoning_core.tasks.logic_derivation import atom_code, canonical_trace


def unary_theory(facts, rules):
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
    )


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


def test_negative_atoms_use_compact_bang_notation():
    assert atom_code(Atom("flagged", ("alice",), False)) == "!flagged(alice)"


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
    assert ex.prompt.startswith("Premise:\n0: ")
    assert ex.prompt.endswith(
        "Give derivation lines as Rule: Input... => Deduction.\n"
        "Use premise IDs and @0, @1, ... for derived lines."
    )

    spaced = ex.answer.replace(":", " : ").replace("=>", " => ")
    assert task.score_answer(spaced, ex) == 1
    assert task.score_answer(ex.answer + "\n0: 0 => bogus(alice)", ex) == 0

    for i, line in enumerate(ex.answer.splitlines()):
        refs = [token for token in line.split() if token.startswith("@")]
        assert all(int(ref[1:]) < i for ref in refs)


def test_logic_derivation_scoring_focuses_on_rule_and_inputs():
    task = get_task("logic_derivation")
    entry = type("Entry", (), {
        "answer": "2: 0 1 => active(alice)\n3: @0 => trusted(alice)"
    })()

    assert task.score_answer(
        "Rule [2]: premises 0, 1 -> Alice becomes active.\n"
        "3: input @0 → Therefore Alice is trusted.",
        entry,
    ) == 1
    assert task.score_answer(
        "Rule 2: Input 0, 1 => Alice becomes active.\n"
        "Rule: 3: Input @0 => Therefore Alice is trusted.",
        entry,
    ) == 1
    assert task.score_answer(
        "2: 0 => Alice becomes active.\n3: @0 => Alice is trusted.",
        entry,
    ) == 0
    assert task.score_answer("2: 0 1 =>\n3: @0 => trusted", entry) == 0
