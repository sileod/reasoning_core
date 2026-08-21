import ast
import statistics
import time

import pytest

from reasoning_core.resources.imperative_mesopy import (
    CONTROLLED_PHENOMENA,
    OBSERVED_ERRORS,
    PHENOMENA,
    ImperativeMesopy,
    MesopyComplexity,
    MesopyConfig,
    structural_fingerprint,
)


def test_execution_is_runnable_and_measured():
    for seed in range(100):
        sample = ImperativeMesopy(seed=seed).execution()
        assert sample.call.ok, (seed, sample.call.error, sample.code)
        assert sample.features["dynamic_steps"] > 0
        assert sample.features["dynamic_lines"] > 0
        assert sample.features["live_fraction"] >= 0.35
        assert sample.features["backward_slice_depth"] > 0
        assert sample.features["param_sensitivity"] >= 0


def test_runnability_is_observed_from_same_program_distribution():
    differing_positions = set()
    errors = set()
    for seed in range(80):
        sample = ImperativeMesopy(seed=seed).runnability_pair()
        assert {call.ok for call in sample.calls} == {True, False}
        assert "haz" not in {n.id for n in ast.walk(ast.parse(sample.code)) if isinstance(n, ast.Name)}
        a, b = sample.calls
        differing_positions.update(i for i, pair in enumerate(zip(a.args, b.args)) if pair[0] != pair[1])
        errors.update(call.error for call in sample.calls if call.error)
    assert len(differing_positions) >= 2
    assert len(errors & set(OBSERVED_ERRORS)) >= 3


def test_optional_alpha_renaming_preserves_semantics_and_fingerprint():
    for seed in range(30):
        plain = ImperativeMesopy(seed=seed).execution(anonymize_names=False)
        anonymous = ImperativeMesopy(seed=seed).execution(anonymize_names=True)
        assert plain.args == anonymous.args
        assert (plain.call.ok, plain.call.value, plain.call.error) == (
            anonymous.call.ok,
            anonymous.call.value,
            anonymous.call.error,
        )
        assert plain.entrypoint == "endpoint"
        assert anonymous.entrypoint != "endpoint"
        assert plain.fingerprint == anonymous.fingerprint
        assert structural_fingerprint(plain.code) == structural_fingerprint(anonymous.code)


def test_complexity_is_recursively_productive_and_measured():
    medians = []
    for level in (0, 3, 6):
        samples = [
            ImperativeMesopy(
                MesopyConfig(complexity=MesopyComplexity.level(level)), seed=seed
            ).execution()
            for seed in range(12)
        ]
        medians.append({
            key: statistics.median(sample.features[key] for sample in samples)
            for key in (
                "ast_nodes",
                "ast_depth",
                "control_depth",
                "call_depth",
                "backward_slice_depth",
                "dynamic_steps",
            )
        })

    assert medians[0]["ast_nodes"] < medians[1]["ast_nodes"] < medians[2]["ast_nodes"]
    assert medians[0]["ast_depth"] < medians[2]["ast_depth"]
    assert medians[0]["control_depth"] < medians[2]["control_depth"]
    assert medians[0]["call_depth"] < medians[2]["call_depth"]
    assert medians[0]["backward_slice_depth"] < medians[2]["backward_slice_depth"]
    assert medians[0]["dynamic_steps"] < medians[1]["dynamic_steps"] < medians[2]["dynamic_steps"]


def test_controlled_phenomena_are_a_subset_not_the_program_skeleton():
    assert set(CONTROLLED_PHENOMENA) < set(PHENOMENA)
    for phenomenon in PHENOMENA:
        for seed in range(20):
            try:
                sample = ImperativeMesopy(seed=seed).execution(
                    phenomena=(phenomenon,),
                    require_recursion=phenomenon == "recursion",
                )
                break
            except RuntimeError:
                continue
        else:
            pytest.fail(f"could not generate requested phenomenon {phenomenon}")
        assert sample.call.ok
        assert phenomenon in sample.phenomena


def test_recursion_is_real_and_terminating():
    sample = ImperativeMesopy(seed=4).execution(require_recursion=True)
    tree = ast.parse(sample.code)
    recursive = []
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == fn.name
            for node in ast.walk(fn)
        ):
            recursive.append(fn)
    assert recursive
    assert sample.call.ok


def test_minimal_pair_uses_step_limited_validation():
    successes = 0
    for seed in range(12):
        sample = ImperativeMesopy(seed=seed).execution()
        try:
            pair = ImperativeMesopy(seed=1000 + seed).minimal_pair(sample, attempts=32)
        except RuntimeError:
            continue
        assert pair.outcome.ok
        assert pair.outcome.value != sample.call.value
        assert pair.outcome.steps is not None
        successes += 1
    assert successes >= 8


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

    cfg = MesopyConfig(complexity=MesopyComplexity.level(6))
    t0 = time.perf_counter()
    for seed in range(8):
        assert ImperativeMesopy(cfg, seed=seed).execution().call.ok
    high_complexity_seconds = time.perf_counter() - t0

    # Wide CI margins. These are regression tripwires, not benchmark claims.
    assert execution_seconds < 5.0
    assert runnability_seconds < 5.0
    assert high_complexity_seconds < 5.0
