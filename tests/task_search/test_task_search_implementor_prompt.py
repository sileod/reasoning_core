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


ROOT = Path(__file__).parents[2]
PLAN = ROOT / "reasoning_core" / "task_search" / "plans" / "wave0.yaml"


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

def test_worker_prompt_combines_global_and_specific_context():
    plan = load_plan(PLAN)
    trial = plan.trials[0]
    metadata = {
        "parent_source_id": None,
        "idea": trial.idea,
        "hypothesis": trial.hypothesis,
        "changes": trial.changes,
        "generation": {"model_name": "example-provider/example-model"},
    }

    prompt = render_implementor_prompt(plan, trial, ROOT, metadata)

    assert "# Agent Notes" in prompt
    assert trial.instruction in prompt
    assert trial.owned_path in prompt
    assert "TASK_META =" in prompt
    assert "example-provider/example-model" in prompt
    assert "samples_N1.md" in prompt
    assert "generate_samples_N1.py" in prompt

def test_sample_generator_command_is_allowed():
    trial = load_plan(PLAN).trials[0]
    permissions = opencode_config(trial, "task-search-worker")["permission"]

    assert permissions["bash"][_sample_command(trial)] == "allow"


def test_prompt_allows_balanced_labels_and_keeps_seeding_outside_tasks():
    plan = load_plan(PLAN)
    prompt = " ".join(render_implementor_prompt(plan, plan.trials[0], ROOT).split())
    assert "Balanced yes/no answers and small fixed label sets are allowed" in prompt
    assert "excess over the 1/k floor" in prompt
    assert "Seed only in the sample script, never inside the task" in prompt
    assert "required before coordinator review" in prompt
    assert "small fixed label sets lose" not in prompt
    assert "seed the `random` module instead" not in prompt

def test_self_check_is_the_only_verification_command_the_prompt_asks_for(tmp_path):
    """The prompt hands out one verification command and the sandbox allows exactly it.

    Trials were spending half a 28-step budget on five separate verification commands,
    and the gates that were not among them -- TASK_META, the contract audit -- only
    surfaced in run.json once the trial was already lost.
    """
    trial = Trial(
        trial_id="N1",
        instruction="Implement it.",
        owned_path="reasoning_core/tasks/generated/wave/example",
        validation=("PYTHONDONTWRITEBYTECODE=1 python -m pytest reasoning_core/tasks",),
        hypothesis="N1",
    )
    command = _selfcheck_command_for(trial.owned_path, trial.trial_id)
    assert command in opencode_permissions(trial)["bash"]
    assert opencode_permissions(trial)["bash"][command] == "allow"

    plan = SearchPlan(
        name="wave",
        base_ref="HEAD",
        context_files=(),
        trials=(trial,),
        queues={},
        path=tmp_path / "plan.yaml",
    )
    plan.path.write_text("version: 1\n")
    prompt = render_implementor_prompt(plan, trial, Path("."))
    assert command in prompt
    # The recipes it replaced are gone: no hand-rolled reproducibility check, no
    # separately quoted prior_audit invocation.
    assert "PYTHONHASHSEED" not in prompt
    assert "prior_audit" not in prompt

def test_prior_audit_sees_a_prompt_that_states_its_own_answer():
    """The gate the wave was missing: eleven mechanical PASSes on a worthless task.

    A generated word problem ended every prompt with the number it was asking for. It
    passed determinism, the contract, pytest and the constant-guess prior -- the answers
    all differ, so nothing keyed on the answer distribution could see it.
    """
    assert prior_audit.shortcuts("gave 3 away, leaving 7 apples.")["last_number"] == "7"
    assert prior_audit.shortcuts("")["last_number"] == ""

    class Copyable:
        """Solvable by reading the last number off the prompt."""

        config = types.SimpleNamespace(set_level=lambda level: None)

        def generate_example(self):
            n = random.randrange(1000)
            return types.SimpleNamespace(
                prompt=f"the total is {n}. What is the total?", answer=str(n)
            )

        def score_answer(self, answer, entry):
            return float(str(answer) == entry.answer)

    task = Copyable()
    report = prior_audit.audit(task, 0, 20, time.time() + 20)
    assert report["const"] < 0.4 and report["distinct"] >= 0.9
    assert report["shortcut"] == 1.0 and report["rule"] == "last_number"

def test_pace_changes_the_prompt_and_nothing_else():
    """The hurry stance is an assumption about the bottleneck, so it has to be A/B-able.

    Only the two pacing strings may differ between arms. Substituting them out has to
    leave three byte-identical prompts, or the arm is confounded with whatever else
    moved -- which is how `pace` leaking into TASK_META was caught.
    """
    plan = load_plan(PLAN)
    trial = plan.trials[0]
    prompts = {
        name: render_implementor_prompt(plan, trial, ROOT, pace=name) for name in PACE
    }

    assert len(set(prompts.values())) == len(PACE)
    assert "Hurry" in prompts["hurry"] and "Hurry" not in prompts["deliberate"]
    assert "two or three formulations" in prompts["deliberate"]

    def without_pacing(text, name):
        # Normalise first: textwrap re-flows the pacing sentence into the surrounding
        # paragraph, so the phrase is only findable once the line breaks are gone.
        text = " ".join(text.split())
        for phrase in PACE[name].values():
            text = text.replace(" ".join(phrase.split()), "<PACING>")
        return text

    stripped = {without_pacing(text, name) for name, text in prompts.items()}
    assert len(stripped) == 1, "pace changed something outside the pacing block"

    # Recorded at the wave level, never inside the provenance mapping the worker pastes.
    assert "pace" not in generation_metadata("m", "v", "a")["settings"]
