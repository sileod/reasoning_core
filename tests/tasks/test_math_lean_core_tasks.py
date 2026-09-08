from reasoning_core.template import Problem, edict
from reasoning_core.tasks import math_lean as ml


def _fake_compile_pair():
    return edict(
        kind="proof_attempt:prop",
        header="theorem ex (p q : Prop) (h0 : p → q) : p → q := by\n",
        positive="intro hp\nexact h0 hp",
        negative="intro hp\nexact hp",
        sampled_attempts=2,
        use_mathlib=False,
    )


def test_candidate_compilation_generate_returns_problem(monkeypatch):
    monkeypatch.setattr(ml, "make_compilation_pair", lambda config: _fake_compile_pair())
    task = ml.LeanCandidateCompilation(ml.LeanConfig(use_mathlib=False))

    ex = task.generate()

    assert isinstance(ex, Problem)
    assert ex.answer in {"A", "B"}
    assert len(ex.answer) == 1
    assert ex.metadata.options["AB".index(ex.answer)] == _fake_compile_pair().positive


def test_candidate_compilation_version_bumped():
    assert ml.LeanCandidateCompilation.task_version == 2


def test_current_lean_tasks_are_registered_and_removed_tasks_stay_removed():
    from reasoning_core import DATASETS

    assert {"lean_missing_line", "lean_candidate_compilation"} <= set(DATASETS)
    assert {
        "lean_compile_selection_indices",
        "lean_derivation_premise_selection",
        "lean_forward_premise_selection",
    }.isdisjoint(DATASETS)
