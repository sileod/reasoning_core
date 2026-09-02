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

def test_bubblewrap_makes_only_owned_path_and_runtime_writable(tmp_path):
    # Strict sandboxes intentionally hide host /tmp, so place this integration
    # fixture on the same non-/tmp filesystem used by real task-search runs.
    with tempfile.TemporaryDirectory(prefix=".task-search-test-", dir=ROOT) as root:
        root = Path(root)
        worktree = root / "worktree"
        owned = worktree / "owned"
        runtime = root / "runtime"
        owned.mkdir(parents=True)
        (worktree / "sibling.txt").write_text("original\n")
        host_pid = __import__("os").getpid()
        with tempfile.NamedTemporaryFile(prefix="task-search-", dir="/tmp") as sentinel:
            command = _sandbox_command(
                [
                    "/bin/bash",
                    "-c",
                    "printf allowed > owned/result.txt; "
                    "if printf forbidden > sibling.txt 2>/dev/null; then exit 9; fi; "
                    f"test ! -e /proc/{host_pid}; "
                    f"test ! -e {sentinel.name}",
                ],
                worktree=worktree,
                owned_path="owned",
                runtime_root=runtime,
            )

            import subprocess

            subprocess.run(command, check=True)

        assert (owned / "result.txt").read_text() == "allowed"
        assert (worktree / "sibling.txt").read_text() == "original\n"

def test_bubblewrap_can_overlay_one_harness_runtime_file():
    with tempfile.TemporaryDirectory(prefix=".task-search-test-", dir=ROOT) as root:
        root = Path(root)
        worktree = root / "worktree"
        owned = worktree / "owned"
        runtime = root / "runtime"
        target = root / "harness-home" / "bin" / "helper"
        source = runtime / "helper"
        owned.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        runtime.mkdir()
        target.write_text("host original\n")
        source.write_text("runtime original\n")
        command = _sandbox_command(
            ["/bin/bash", "-c", f"printf worker > {target}"],
            worktree=worktree,
            owned_path="owned",
            runtime_root=runtime,
            writable_overlays=((source, target),),
        )

        subprocess.run(command, check=True)

        assert source.read_text() == "worker"
        assert target.read_text() == "host original\n"

def test_systemd_resource_wrapper_records_hard_limits():
    command = _resource_command(
        ["bwrap", "true"],
        {
            "enabled": True,
            "executable": "/usr/bin/systemd-run",
            "memory_max": "8G",
            "tasks_max": 512,
            "cpu_quota": "400%",
        },
    )

    assert "MemoryMax=8G" in command
    assert "TasksMax=512" in command
    assert "CPUQuota=400%" in command
    assert command[-2:] == ["bwrap", "true"]

def test_independent_validation_times_out(tmp_path):
    with tempfile.TemporaryDirectory(prefix=".task-search-test-", dir=ROOT) as root:
        root = Path(root)
        worktree = root / "worktree"
        owned = worktree / "owned"
        owned.mkdir(parents=True)
        results = _run_validation(
            worktree,
            ("/bin/sleep 2",),
            root / "validation.log",
            owned_path="owned",
            runtime_root=root / "runtime",
            bwrap_bin="bwrap",
            resource_limits={"enabled": False},
            timeout_seconds=0.05,
        )

    assert results == [
        {
            "command": "/bin/sleep 2",
            "exit_code": 124,
            "timed_out": True,
        }
    ]

def test_validation_environment_removes_named_credentials(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "secret")
    monkeypatch.setenv("KEEP_ME", "visible")

    environment = _sanitized_environment(("PROVIDER_API_KEY",))

    assert "PROVIDER_API_KEY" not in environment
    assert environment["KEEP_ME"] == "visible"

def test_sandboxed_validation_does_not_receive_named_credential(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "secret")
    with tempfile.TemporaryDirectory(prefix=".task-search-test-", dir=ROOT) as root:
        root = Path(root)
        worktree = root / "worktree"
        (worktree / "owned").mkdir(parents=True)
        results = _run_validation(
            worktree,
            ('test -z "${PROVIDER_API_KEY+x}"',),
            root / "validation.log",
            owned_path="owned",
            runtime_root=root / "runtime",
            bwrap_bin="bwrap",
            resource_limits={"enabled": False},
            timeout_seconds=5,
            credential_env_names=("PROVIDER_API_KEY",),
        )

    assert results[0]["exit_code"] == 0
