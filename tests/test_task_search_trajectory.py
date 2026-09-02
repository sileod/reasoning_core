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

def test_live_trajectory_reads_budget_from_invocation_summary(tmp_path):
    wave = tmp_path / "run"
    trial = wave / "S26"
    trial.mkdir(parents=True)
    (wave / "summary.json").write_text(json.dumps({"max_steps": 28}))
    (trial / "events.jsonl").write_text(
        json.dumps({"type": "step_start", "part": {}}) + "\n"
    )

    row = trajectory.read(trial)

    assert row["status"] == "no run.json"
    assert row["steps"] == 1
    assert row["budget"] == 28

def test_trajectory_marks_a_truncated_selfcheck_incomplete(tmp_path):
    trial = tmp_path / "run" / "S35"
    trial.mkdir(parents=True)
    event = {
        "type": "tool_use",
        "part": {
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {
                    "command": "python -m reasoning_core.task_search.selfcheck owned S35"
                },
                "output": "implementation PASS\nspeed PASS\n<shell_metadata>timed out</shell_metadata>",
            },
        },
    }
    (trial / "events.jsonl").write_text(json.dumps(event) + "\n")

    row = trajectory.read(trial)

    assert row["checks"][-1]["incomplete"] == "FAIL"

def test_trajectory_reads_agy_stream_json(tmp_path):
    trial = tmp_path / "run" / "S41"
    trial.mkdir(parents=True)
    events = [
        {
            "event": "step_update",
            "step_update": {"step_type": "agent_response", "state": "DONE"},
        },
        {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "state": "DONE",
                "tool_name": "run_command",
                "tool_info": {
                    "parameters": {
                        "CommandLine": "python -m reasoning_core.task_search.selfcheck owned S41"
                    },
                    "output": "implementation PASS\n0 gate(s) failing.\n",
                },
            },
        },
        {"event": "result", "result": {"status": "SUCCESS"}},
    ]
    (trial / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )

    row = trajectory.read(trial)

    assert row["steps"] == 1
    assert row["stopped"] == "SUCCESS"
    assert row["checks"] == [{"implementation": "PASS"}]

def test_step_usage_flags_a_worker_that_ran_out_of_budget(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        "".join(json.dumps({"type": "step_start"}) + "\n" for _ in range(28))
    )
    assert _step_usage(events, 28) == {"used": 28, "max": 28, "exhausted": True}
    assert _step_usage(events, 60)["exhausted"] is False
    assert _step_usage(tmp_path / "absent.jsonl", 28) is None

def test_step_usage_understands_agy_stream_json(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {"step_type": "agent_response", "state": "DONE"},
                    }
                ),
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {"step_type": "tool", "state": "DONE"},
                    }
                ),
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {"step_type": "agent_response", "state": "DONE"},
                    }
                ),
            ]
        )
    )

    assert _step_usage(events, None) == {"used": 2, "max": None, "exhausted": False}
