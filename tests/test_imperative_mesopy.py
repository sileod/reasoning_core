import ast
import statistics
import time

from reasoning_core.resources.imperative_mesopy import (
    ERRORS,
    PHENOMENA,
    ImperativeMesopy,
    MesopyComplexity,
    MesopyConfig,
)


def test_execution_is_runnable_by_construction():
    for seed in range(100):
        sample = ImperativeMesopy(seed=seed).execution()
        assert sample.call.ok, (seed, sample.call.error, sample.code)
        compile(sample.code, "<test-imperative-mesopy>", "exec")


def test_safe_hazards_also_appear_in_successful_programs():
    samples = [ImperativeMesopy(seed=seed).execution() for seed in range(80)]
    hazardous = [sample for sample in samples if sample.features["hazard"] is not None]
    assert len(hazardous) >= 15
    assert all(sample.call.ok for sample in hazardous)
    assert len({sample.features["hazard"] for sample in hazardous}) >= 3


def test_all_controlled_phenomena_and_recursion_are_supported():
    for phenomenon in PHENOMENA:
        sample = ImperativeMesopy(seed=7).execution(
            phenomena=(phenomenon,),
            require_recursion=phenomenon == "recursion",
        )
        assert sample.call.ok, (phenomenon, sample.call.error, sample.code)
        assert phenomenon in sample.phenomena

    sample = ImperativeMesopy(seed=9).execution(require_recursion=True)
    tree = ast.parse(sample.code)
    rec = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "rec"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "rec"
        for node in ast.walk(rec)
    )


def test_runnability_pairs_use_identical_source_with_opposite_outcomes():
    for error in ERRORS:
        for seed in range(8):
            sample = ImperativeMesopy(seed=seed).runnability_pair(error=error)
            assert {call.ok for call in sample.calls} == {True, False}
            assert any(call.error == error for call in sample.calls)


def test_complexity_budgets_are_structurally_productive():
    medians = []
    for level in (0, 3, 6):
        features = [
            ImperativeMesopy(
                MesopyConfig(complexity=MesopyComplexity.level(level)),
                seed=seed,
            ).execution().features
            for seed in range(12)
        ]
        medians.append({
            key: statistics.median(sample[key] for sample in features)
            for key in (
                "ast_nodes",
                "ast_depth",
                "control_depth",
                "call_depth",
                "dataflow_depth",
            )
        })

    assert medians[0]["ast_nodes"] < medians[1]["ast_nodes"] < medians[2]["ast_nodes"]
    assert medians[0]["ast_depth"] < medians[2]["ast_depth"]
    assert medians[0]["control_depth"] < medians[2]["control_depth"]
    assert medians[0]["call_depth"] < medians[2]["call_depth"]
    assert medians[0]["dataflow_depth"] < medians[2]["dataflow_depth"]


def test_profiling_is_opt_in_and_preserves_outcome():
    generator = ImperativeMesopy(seed=13)
    sample = generator.execution()
    profiled = generator.profile(sample)
    assert profiled.ok == sample.call.ok
    assert profiled.value == sample.call.value
    assert profiled.steps > 0
    assert profiled.elapsed >= 0


def test_generation_throughput_stays_fast():
    t0 = time.perf_counter()
    for seed in range(100):
        assert ImperativeMesopy(seed=seed).execution().call.ok
    execution_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    for seed in range(50):
        sample = ImperativeMesopy(seed=seed).runnability_pair()
        assert {call.ok for call in sample.calls} == {True, False}
    runnability_seconds = time.perf_counter() - t0

    assert execution_seconds < 3.0
    assert runnability_seconds < 3.0
