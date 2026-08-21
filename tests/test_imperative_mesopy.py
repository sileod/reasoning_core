from reasoning_core.resources.imperative_mesopy import (
    ImperativeMesopy,
    MesopyGoal,
)


CONTROLLED_PHENOMENA = (
    "aliasing",
    "closure_late_binding",
    "default_capture",
    "mutation_call",
    "loop_carried_state",
    "rebinding_vs_aliasing",
)


def test_imperative_mesopy_execution_is_runnable():
    for seed in range(20):
        sample = ImperativeMesopy(seed=seed).execution()
        assert sample.call.ok
        compile(sample.code, "<test-imperative-mesopy>", "exec")
        assert sample.features["ast_nodes"] > 30
        assert sample.features["dataflow_depth"] >= 4


def test_imperative_mesopy_supersets_controlled_phenomena():
    goal = MesopyGoal(
        phenomena=CONTROLLED_PHENOMENA,
        min_phenomena=len(CONTROLLED_PHENOMENA),
        max_phenomena=8,
    )
    for seed in range(10):
        sample = ImperativeMesopy(seed=seed).generate(goal)
        assert sample.call.ok
        assert set(CONTROLLED_PHENOMENA) <= set(sample.phenomena)


def test_runnability_pair_uses_identical_code_with_opposite_outcomes():
    for error in ("IndexError", "ZeroDivisionError", "ValueError"):
        for seed in range(10):
            sample = ImperativeMesopy(seed=seed).runnability_pair(error=error)
            assert len(sample.calls) == 2
            assert {call.ok for call in sample.calls} == {True, False}
            assert any(call.error == error for call in sample.calls)


def test_requested_failure_is_generated_by_semantics():
    for error in ("IndexError", "ZeroDivisionError", "ValueError"):
        sample = ImperativeMesopy(seed=3).generate(
            MesopyGoal(runnable=False, error=error)
        )
        assert not sample.call.ok
        assert sample.call.error == error


def test_surface_and_semantic_composition_are_diverse():
    samples = [ImperativeMesopy(seed=seed).execution() for seed in range(20)]
    assert len({sample.code for sample in samples}) >= 18
    assert len({sample.phenomena for sample in samples}) >= 12
