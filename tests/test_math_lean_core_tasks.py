from reasoning_core.template import Problem, edict
from reasoning_core.tasks import math_lean as ml


def _fake_compile_instance():
    return edict(
        kind="core_prop_chain",
        header="theorem ex (p q : Prop) (h0 : p → q) : p → q := by\n",
        candidates=["rfl", "exact h0", "intro hp; exact h0 hp", "simp"],
        labels=[False, True, True, False],
        primary="exact h0",
        elegant="exact h0",
        use_mathlib=False,
    )


def _fake_nondiscriminative_instance():
    return edict(
        kind="poly_eq",
        header="theorem ex (a : Int) : a + 0 = a := by\n",
        candidates=["ring", "simp", "rfl"],
        labels=[True, True, False],
        primary="ring",
        elegant="ring",
        use_mathlib=True,
    )


def test_candidate_compilation_generate_returns_problem(monkeypatch):
    monkeypatch.setattr(ml, "make_instance", lambda config: _fake_compile_instance())
    task = ml.LeanCandidateCompilation(ml.LeanConfig(use_mathlib=False))

    ex = task.generate()

    assert isinstance(ex, Problem)
    assert ex.answer in {"True", "False"}
    assert len(ex.answer) <= 5


def test_candidate_compilation_does_not_require_discriminative_selection(monkeypatch):
    monkeypatch.setattr(ml, "make_instance", lambda config: _fake_nondiscriminative_instance())
    task = ml.LeanCandidateCompilation(ml.LeanConfig(use_mathlib=True))

    ex = task.generate()

    assert isinstance(ex, Problem)
    assert ex.metadata.kind == "poly_eq"


def test_current_lean_tasks_are_registered_and_removed_tasks_stay_removed():
    from reasoning_core import DATASETS

    assert {"lean_missing_line", "lean_candidate_compilation"} <= set(DATASETS)
    assert {
        "lean_compile_selection_indices",
        "lean_derivation_premise_selection",
        "lean_forward_premise_selection",
    }.isdisjoint(DATASETS)
