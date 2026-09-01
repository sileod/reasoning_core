"""Legacy import and proposal-to-plan fan-out."""

from pathlib import Path

import pytest
import yaml

from reasoning_core.task_search.legacy import build_legacy_wave, read_legacy_candidates
from reasoning_core.task_search.plan import load_plan
from reasoning_core.task_search.plan_builder import build_plan, write_plan
from reasoning_core.task_search.wave_proposer import (
    proposal_problems,
    validate_proposal_wave,
)


ROOT = Path(__file__).parents[1]


def wave(count=2, variants_of=("alpha_task", "beta_task")):
    return {
        "kind": "sft_task_proposals",
        "name": "wave0",
        "proposals": [
            {"id": f"P{index:03d}", "name": name,
             "summary": f"Generate {name} instances over varied structures and answer one value."}
            for index, name in enumerate(variants_of[:count], 1)
        ],
    }


def test_legacy_descriptions_are_already_coverage_summaries():
    candidates = read_legacy_candidates(ROOT)

    assert len(candidates) == 80
    assert ("strongly_connected_component",
            "Given a directed graph and node, output the sorted members of its strongly"
            " connected component.", "W1-001") in candidates
    for name, summary, _ in candidates:
        assert proposal_problems({"name": name, "summary": summary}) == []


def test_legacy_wave_imports_without_a_model_call():
    imported = build_legacy_wave(ROOT)

    assert imported["name"] == "wave0"
    assert imported["generation"]["provider"] == "legacy"
    assert imported["generation"]["calls"] == []
    assert len(imported["proposals"]) == 80
    assert imported["rejected"] == []
    assert validate_proposal_wave(imported) == []


def test_an_imported_wave_does_not_claim_a_novelty_verdict_it_never_got():
    """The reference wave was never reviewed; the record says so rather than inventing scores."""
    novelty = build_legacy_wave(ROOT)["proposals"][0]["novelty"]

    assert novelty["source"] == "legacy"
    assert novelty["verdict"] == "imported"
    assert "scores" not in novelty


def test_one_proposal_fans_out_into_independent_draws():
    plan = build_plan(wave(count=1), name="wave9", base_ref="abc123", variants=3)

    assert [trial["id"] for trial in plan["trials"]] == ["P001v1", "P001v2", "P001v3"]
    assert len({trial["owned_path"] for trial in plan["trials"]}) == 3
    # Draws differ only in the task name they must register under. Same summary, same
    # guidance, and the runner seeds each trial from its id, so what varies is sampling.
    generic = {trial["instruction"].replace(f"alpha_task_v{index}", "alpha_task")
               for index, trial in enumerate(plan["trials"], 1)}
    assert len(generic) == 1
    assert plan["queues"]["v2"] == ["P001v2"]


def test_a_single_draw_keeps_the_plain_task_name():
    plan = build_plan(wave(count=1), name="wave9", variants=1)

    assert plan["trials"][0]["owned_path"] == (
        "reasoning_core/tasks/generated/wave9/alpha_task")


def test_a_generated_plan_loads_in_the_runner(tmp_path):
    path = tmp_path / "wave9.yaml"
    write_plan(path, build_plan(wave(), name="wave9", base_ref="abc123", variants=2))

    loaded = load_plan(path)

    assert loaded.name == "wave9"
    assert loaded.base_ref == "abc123"
    assert len(loaded.trials) == 4
    assert set(loaded.queues) == {"v1", "v2", "pilot"}
    assert yaml.safe_load(path.read_text())["proposal_wave"] == "wave0"


def test_a_plan_name_must_be_an_importable_package_segment():
    """owned_path becomes a module path: a dash there fails the contract audit, not the run."""
    with pytest.raises(ValueError, match="importable|identifier"):
        build_plan(wave(), name="wave0-run1")


def test_plan_generation_refuses_a_wave_that_is_not_a_proposal_wave():
    with pytest.raises(ValueError, match="proposal wave"):
        build_plan({"kind": "search_plan", "proposals": []}, name="wave9")
