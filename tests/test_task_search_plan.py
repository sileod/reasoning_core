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
from reasoning_core.task_search.runner import (
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

    wave8 was two implementation runs of the eighty wave0 proposals. The plan said so and
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
    plan.write_text("version: 1\nname: wave0_r2\nproposal_wave: wave0\n" + body)
    assert load_plan(plan).proposal_wave == "wave0"
    assert load_plan(plan).name == "wave0_r2"

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
