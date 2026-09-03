import dataclasses
import json
import random
import time
import types
from pathlib import Path
import subprocess
import tempfile

import pytest

from reasoning_core.task_search import prior_audit, trajectory, validation

from reasoning_core.task_search.implementor_prompt import (
    PACE,
    _prior_audit_command,
    _sample_command,
    _sample_command_for,
    _selfcheck_command_for,
    render_implementor_prompt,
)
from reasoning_core.task_search.plan import (
    SearchPlan,
    Trial,
    _frozen_module_drift,
    _plan_problems,
    _select_trials,
    load_plan,
)
from reasoning_core.task_search.implementation_runner import (
    _mini_config,
    _prepare_harness,
    _retryable_harness_failure,
    generation_metadata,
    opencode_config,
    opencode_permissions,
)
from reasoning_core.task_search.sandbox import (
    _resource_command,
    _run_validation,
    _sandbox_command,
    _sanitized_environment,
)
from reasoning_core.task_search.validation import (
    _outside_owned,
    _owned_digest,
    _review_source,
    _sample_review,
    _sample_sanity,
    _step_usage,
    _task_classes,
    _task_metadata,
    _undiscoverable,
    sample_shortfall,
)


ROOT = Path(__file__).parents[1]
PLAN = ROOT / "reasoning_core" / "task_search" / "wave0.yaml"


# A prompt of realistic length, because the gate now measures prompt text as well as
# answers: a file of headings with nothing under them is not a worked example.
SAMPLE_PROMPT = (
    "Prompt:\nSort the multiset {3, 1, 2} and report the median as an"
    " integer. Show the sorted order first, then the element that sits"
    " in the middle of it.\n"
)
SAMPLE_BODY = "".join(
    f"# Level {level}\n" + (SAMPLE_PROMPT + f"Answer: {level}{index}\n") * 2
    for index, level in enumerate(("0", "2", "5"))
)

def test_wave0_plan_is_valid_and_folder_scoped():
    plan = load_plan(PLAN)

    assert plan.name == "WAVE0"
    assert len(plan.trials) == 32
    assert len({trial.owned_path for trial in plan.trials}) == 32
    assert all(
        trial.owned_path.startswith("reasoning_core/tasks/generated/wave0/")
        for trial in plan.trials[:12]
    )
    assert all(
        trial.owned_path.startswith("reasoning_core/tasks/mutated/wave0/")
        for trial in plan.trials[12:]
    )
    assert plan.queues["pilot"] == ("N4", "M1")
    assert len(plan.queues["weekend_p0"]) == 17

def test_queue_and_explicit_trials_are_combined_in_plan_order():
    plan = load_plan(PLAN)

    selected = _select_trials(plan, ("N2",), ("pilot",))

    assert [trial.trial_id for trial in selected] == ["N2", "N4", "M1"]

def test_plan_rejects_overlapping_owned_paths(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "version: 1\n"
        "name: overlap\n"
        "trials:\n"
        "  - id: A\n"
        "    idea: a\n"
        "    changes: a\n"
        "    instruction: a\n"
        "    owned_path: out/a\n"
        "    validation: [check-a]\n"
        "  - id: B\n"
        "    idea: b\n"
        "    changes: b\n"
        "    instruction: b\n"
        "    owned_path: out/a/nested\n"
        "    validation: [check-b]\n"
    )

    with pytest.raises(ValueError, match="owned paths overlap"):
        load_plan(plan)

def test_a_plan_records_which_proposal_wave_it_implements(tmp_path):
    """One set of ideas can be implemented many times, so the two names are not one name.

    wave8 was two implementation runs of the eighty external proposals. The plan said so and
    load_plan dropped it, so every run record claimed only `wave: wave8` and comparing one
    proposal wave against another meant remembering by hand which plan came from where.
    """
    body = (
        "trials:\n"
        "  - id: A\n"
        "    idea: a\n"
        "    changes: a\n"
        "    instruction: a\n"
        "    owned_path: out/a\n"
        "    validation: [check-a]\n"
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text("version: 1\nname: external_r2\nproposal_wave: external\n" + body)
    assert load_plan(plan).proposal_wave == "external"
    assert load_plan(plan).name == "external_r2"

    # Plans written before the field existed stay loadable and report it as unrecorded.
    older = tmp_path / "older.yaml"
    older.write_text("version: 1\nname: wave7\n" + body)
    assert load_plan(older).proposal_wave == ""


def test_plan_problems_are_the_ones_check_used_to_miss():
    """A plan could pass `check` and still have nowhere to run.

    Each of these surfaced only at launch, after the worktrees were made: an owned path
    the contract audit cannot turn into an import, a missing prompt context would
    fail to read, and a base_ref that names nothing.
    """
    plan = load_plan(PLAN)
    assert _plan_problems(plan, ROOT) == []

    misplaced = dataclasses.replace(
        plan,
        context_files=("no/such/guide.md",),
        trials=(dataclasses.replace(plan.trials[0], owned_path="scratch/n1"),),
    )
    problems = _plan_problems(misplaced, ROOT)
    assert any("context file missing" in problem for problem in problems)
    assert any("outside reasoning_core/tasks" in problem for problem in problems)

    unresolvable = dataclasses.replace(plan, base_ref="no-such-ref")
    assert _plan_problems(unresolvable, ROOT) == [
        "base_ref does not resolve to a commit: no-such-ref"
    ]

def test_frozen_module_drift_catches_a_base_ref_left_behind(tmp_path):
    """Workers run the harness modules frozen at base_ref; the gates are whatever is live.

    Nothing else in the harness compares those two, so a gate tightened without moving
    base_ref forward would go out silently -- and the worker it fails would have been
    told, by the harness itself, that it had passed. A flag added to prior_audit is
    worse still: the coordinator writes the command line live, so the pinned copy is
    handed an argument it has never heard of.
    """
    paths = {
        name: tmp_path / f"reasoning_core/task_search/{name}.py"
        for name in (
            "selfcheck",
            "validation",
            "sandbox",
            "implementor_prompt",
            "prior_audit",
        )
    }
    git = lambda *args: subprocess.run(
        ("git",) + args, cwd=tmp_path, check=True, capture_output=True
    )
    git("init", "-q")
    git("config", "user.email", "t@t"), git("config", "user.name", "t")
    (tmp_path / "unrelated").write_text("x\n")
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    git("add", "unrelated"), git("commit", "-qm", "before the harness existed")
    assert "cannot run it at all" in _frozen_module_drift(tmp_path, "HEAD")

    git("add", "-A"), git("commit", "-qm", "pin them")
    assert _frozen_module_drift(tmp_path, "HEAD") == ""

    # A flag added to prior_audit alone is enough: the pinned copy would reject it.
    paths["prior_audit"].write_text("# prior_audit --max-shortcut\n")
    drift = _frozen_module_drift(tmp_path, "HEAD")
    assert "Move base_ref forward" in drift and "prior_audit.py" in drift


def test_a_trial_without_a_design_choice_renders_the_prompt_it_always_did(tmp_path):
    """The design-choice section is built but not in use, so it must cost nothing.

    Every wave so far fanned its variants on seed alone. Adding an unused field that
    shifted a single prompt byte would make those waves incomparable with the next one
    for no gain, so the empty case renders no section at all.
    """
    def plan_with(extra):
        path = tmp_path / f"{'set' if extra else 'unset'}.yaml"
        path.write_text(
            "version: 1\nname: w\nproposal_wave: p\ntrials:\n"
            "  - id: A\n    idea: a\n    changes: a\n    instruction: Implement it.\n"
            "    owned_path: reasoning_core/tasks/generated/w/a\n"
            "    validation: [check-a]\n" + extra
        )
        return load_plan(path)

    unset = plan_with("")
    assert unset.trials[0].design_choice == ""
    without = render_implementor_prompt(unset, unset.trials[0], Path.cwd())
    assert "design choice" not in without.lower()

    chosen = plan_with("    design_choice: Answer with the witness, not the verdict.\n")
    assert chosen.trials[0].design_choice == "Answer with the witness, not the verdict."
    with_choice = render_implementor_prompt(chosen, chosen.trials[0], Path.cwd())
    assert "## Assigned design choice" in with_choice
    assert "Answer with the witness, not the verdict." in with_choice
    # The only difference is the new section: nothing else about the prompt moved.
    assert without == with_choice.replace(
        with_choice[with_choice.index("\n## Assigned design choice") :
                    with_choice.index("\nDesign constraint, measured on this wave:")],
        "",
    )


def test_design_choices_fan_variants_across_approaches_not_seeds():
    """One choice per variant, or the wave reports one approach as two draws."""
    from reasoning_core.task_search.plan_builder import build_plan

    wave = {
        "kind": "sft_task_proposals",
        "name": "p",
        "proposals": [{"id": "P001", "name": "thing", "summary": "Do a thing."}],
    }
    built = build_plan(wave, name="w", variants=2,
                       design_choices={"P001": ("choice one", "choice two")})
    assert [t["design_choice"] for t in built["trials"]] == ["choice one", "choice two"]

    # Unused, the field is absent rather than empty: an old plan and a new one with no
    # choices are the same bytes.
    plain = build_plan(wave, name="w", variants=2)
    assert all("design_choice" not in trial for trial in plain["trials"])

    with pytest.raises(ValueError, match="have to match"):
        build_plan(wave, name="w", variants=2, design_choices={"P001": ("only one",)})


def test_the_design_proposer_rejects_a_short_or_duplicated_reply():
    """Fewer distinct choices than variants would run one approach twice."""
    from reasoning_core.task_search import design_proposer

    class Client:
        def __init__(self, choices):
            self.choices = choices

        def json(self, _label, _system, _prompt):
            return {"choices": self.choices}

    assert design_proposer.propose_design_choices(
        "t", "A summary.", 2, client=Client(["  first  way ", "second way"])
    ) == ("first way", "second way")

    with pytest.raises(ValueError, match="got 1"):
        design_proposer.propose_design_choices(
            "t", "A summary.", 2, client=Client(["same way", "SAME WAY"])
        )
    with pytest.raises(ValueError, match="choices list"):
        design_proposer.propose_design_choices(
            "t", "A summary.", 2, client=Client("not a list")
        )
