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
    _RETRY_CEILING_SECONDS,
    _retry_delay,
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

def test_opencode_profile_leaves_write_scope_to_mount_sandbox():
    trial = load_plan(PLAN).trials[0]
    config = opencode_config(trial, "task-search-worker")
    permissions = config["agent"]["task-search-worker"]["permission"]

    assert permissions["edit"] == "allow"
    assert permissions["bash"]["*"] == "deny"
    assert permissions["bash"][trial.validation[0]] == "allow"
    assert permissions["task"] == "deny"


def test_snapshots_toggle_does_not_change_worker_permissions():
    trial = load_plan(PLAN).trials[0]
    disabled = opencode_config(trial, "worker")
    enabled = opencode_config(trial, "worker", snapshots=True)
    assert disabled.pop("snapshot") is False
    assert enabled.pop("snapshot") is True
    assert disabled == enabled


@pytest.mark.parametrize("runs_root", ["/tmp/task-search-test", "/run/task-search-test"])
def test_hidden_run_directory_fails_before_worktree_or_harness_setup(monkeypatch, runs_root):
    from reasoning_core.task_search import implementation_runner as runner

    def unexpected(*args, **kwargs):
        pytest.fail("invalid run directory reached subprocess setup")

    monkeypatch.setattr(runner.subprocess, "check_output", unexpected)
    with pytest.raises(ValueError, match="runs root cannot be under"):
        runner.run_plan(PLAN, repo_root=ROOT, model="unused", runs_root=runs_root)

def test_generation_metadata_records_requested_but_unforwarded_seed():
    metadata = generation_metadata(
        "example-provider/example-model",
        "1.18.20",
        "task-search-worker",
        requested_seed=42,
    )

    assert metadata["provider_name"] == "example-provider"
    assert metadata["settings"]["requested_seed"] == 42
    assert metadata["settings"]["seed_forwarded"] is False

def test_generation_metadata_records_selected_harness():
    metadata = generation_metadata(
        "example-model",
        "1.15.0",
        "mini-default",
        provider_name="provider-cli",
        harness_name="mini",
    )

    assert metadata["harness_name"] == "mini"
    assert metadata["provider_name"] == "provider-cli"

def test_opencode_profile_forwards_seed_when_requested():
    trial = load_plan(PLAN).trials[0]
    config = opencode_config(
        trial,
        "task-search-worker",
        requested_seed=123,
        forward_seed=True,
        max_steps=17,
    )

    assert config["agent"]["task-search-worker"]["seed"] == 123
    assert config["agent"]["task-search-worker"]["steps"] == 17

def test_hlink_is_the_only_harness_launcher(tmp_path):
    command = _prepare_harness(
        "hlink",
        "opencode",
        worktree=tmp_path,
        prompt="assignment",
        model="example-model",
        provider="nim",
        agent="worker",
        variant=None,
        config_path=None,
        trajectory_path=None,
        timeout_seconds=90,
        agy_log_path=tmp_path / "agy.log",
    )

    assert command[:2] == ["hlink", "opencode"]
    assert command[command.index("-C") + 1] == str(tmp_path)
    assert command[command.index("-p") + 1] == "assignment"
    assert command[command.index("--provider") + 1] == "nim"
    assert command[command.index("--") + 1 :] == [
        "--pure",
        "--agent",
        "worker",
        "--format",
        "json",
    ]

def test_retry_classifier_accepts_explicit_provider_transients_only(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "name": "APIError",
                    "data": {"statusCode": 429, "isRetryable": True},
                },
            }
        )
        + "\n"
    )
    transient = {
        "status": "harness_failed",
        "harness_exit_code": 1,
        "harness_log": str(events),
    }

    assert _retryable_harness_failure(transient) == "provider_429"
    assert (
        _retryable_harness_failure({**transient, "status": "validation_failed"}) is None
    )
    events.write_text(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "name": "ConfigurationError",
                    "data": {"statusCode": 400, "isRetryable": False},
                },
            }
        )
        + "\n"
    )
    assert _retryable_harness_failure(transient) is None
    assert (
        _retryable_harness_failure(
            {"status": "harness_failed", "harness_exit_code": -15}
        )
        == "signal_15"
    )


def test_a_provider_backoff_grows_past_one_minute_and_is_jittered():
    """wave8 burned 120 attempts retrying a shared token bucket in lockstep.

    The delay used to be capped at 60s, which against a per-minute limit meant every
    worker woke into the same saturated minute, every time. Growth is what eventually
    outlasts the bucket; jitter is what stops the herd re-arriving together.
    """
    third = [_retry_delay(60, 3) for _ in range(50)]
    assert min(third) > 60, "the delay must be able to outgrow a one-minute window"
    assert len(set(third)) > 1, "an unjittered delay retries the whole wave in lockstep"
    assert max(third) <= _RETRY_CEILING_SECONDS * 1.5

    assert all(_retry_delay(30, 9) <= _RETRY_CEILING_SECONDS * 1.5 for _ in range(20))
    assert sum(_retry_delay(60, 1) for _ in range(400)) / 400 == pytest.approx(60, rel=0.2)
