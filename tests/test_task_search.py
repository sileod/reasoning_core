import dataclasses
import json
import random
import time
import types
from pathlib import Path
import subprocess
import tempfile

import pytest

from reasoning_core.task_search import prior_audit, selfcheck
from reasoning_core.task_search import trajectory

from reasoning_core.task_search.runner import (
    _sample_command_for,
    _step_usage,
    _sample_sanity,
    _review_source,
    opencode_permissions,
    Trial,
    SearchPlan,
    _selfcheck_command_for,
    _adapter_command,
    _agy_command,
    _harness_version,
    _mini_command,
    _mini_config,
    _outside_owned,
    _opencode_command,
    _resource_command,
    _resolve_harness_executable,
    _resolve_opencode_executable,
    _run_validation,
    _sandbox_command,
    _sample_review,
    _sample_command,
    _sanitized_environment,
    _select_trials,
    _task_classes,
    _task_metadata,
    _owned_digest,
    _plan_problems,
    _frozen_module_drift,
    sample_shortfall,
    _prior_audit_command,
    _undiscoverable,
    generation_metadata,
    render_prompt,
    PACE,
    load_plan,
    opencode_config,
    render_prompt,
)


ROOT = Path(__file__).parents[1]
PLAN = ROOT / "reasoning_core" / "task_search" / "wave0.yaml"


def test_wave0_plan_is_valid_and_folder_scoped():
    plan = load_plan(PLAN)

    assert plan.name == "WAVE0"
    assert len(plan.trials) == 32
    assert len({trial.owned_path for trial in plan.trials}) == 32
    assert all(trial.owned_path.startswith("reasoning_core/tasks/generated/wave0/")
               for trial in plan.trials[:12])
    assert all(trial.owned_path.startswith("reasoning_core/tasks/mutated/wave0/")
               for trial in plan.trials[12:])
    assert plan.queues["pilot"] == ("N4", "M1")
    assert len(plan.queues["weekend_p0"]) == 17


def test_worker_prompt_combines_global_and_specific_context():
    plan = load_plan(PLAN)
    trial = plan.trials[0]
    metadata = {
        "parent_source_id": None,
        "idea": trial.idea,
        "hypothesis": trial.hypothesis,
        "changes": trial.changes,
        "generation": {"model_name": "albert/deepseek-v4-flash"},
    }

    prompt = render_prompt(plan, trial, ROOT, metadata)

    assert "# Agent Notes" in prompt
    assert trial.instruction in prompt
    assert trial.owned_path in prompt
    assert "TASK_META =" in prompt
    assert "albert/deepseek-v4-flash" in prompt
    assert "samples_N1.md" in prompt
    assert "generate_samples_N1.py" in prompt


def test_opencode_profile_leaves_write_scope_to_mount_sandbox():
    trial = load_plan(PLAN).trials[0]
    config = opencode_config(trial, "task-search-worker")
    permissions = config["agent"]["task-search-worker"]["permission"]

    assert permissions["edit"] == "allow"
    assert permissions["bash"]["*"] == "deny"
    assert permissions["bash"][trial.validation[0]] == "allow"
    assert permissions["task"] == "deny"


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
                    "/bin/bash", "-c",
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


def test_generation_metadata_records_requested_but_unforwarded_seed():
    metadata = generation_metadata(
        "albert/deepseek-v4-flash",
        "1.18.20",
        "task-search-worker",
        requested_seed=42,
    )

    assert metadata["provider_name"] == "albert"
    assert metadata["settings"]["requested_seed"] == 42
    assert metadata["settings"]["seed_forwarded"] is False


def test_generation_metadata_records_selected_harness():
    metadata = generation_metadata(
        "deepseek-v4-flash", "1.15.0", "mini-default",
        provider_name="albert", adapter_name="harness-link",
        harness_name="mini",
    )

    assert metadata["harness_name"] == "mini"
    assert metadata["provider_name"] == "albert"


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


def test_prompt_is_a_positional_message_not_a_greedy_file_argument(tmp_path):
    command = _opencode_command(
        "opencode",
        model="albert/deepseek-v4-flash",
        agent="task-search-worker",
        worktree=tmp_path,
        prompt="complete prompt",
    )

    assert "--file" not in command
    assert command[-1] == "complete prompt"


def test_harness_link_wraps_opencode_without_replacing_its_arguments(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda value: "/usr/local/bin/albert")
    direct = ["opencode", "run", "--pure", "prompt"]

    command = _adapter_command(
        direct,
        adapter="harness-link",
        provider="albert",
        model="deepseek-v4-flash",
    )

    assert command == [
        "/usr/local/bin/albert", "opencode", "--model",
        "deepseek-v4-flash", "--", "run", "--pure", "prompt",
    ]


def test_harness_link_wraps_mini_without_replacing_its_arguments(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda value: "/usr/local/bin/albert")
    direct = ["mini", "-c", "mini.yaml", "-t", "prompt"]

    command = _adapter_command(
        direct,
        adapter="harness-link",
        provider="albert",
        model="deepseek-v4-flash",
        harness="mini",
    )

    assert command == [
        "/usr/local/bin/albert", "mini", "--model",
        "deepseek-v4-flash", "--", "-c", "mini.yaml", "-t", "prompt",
    ]


def test_mini_command_and_config_are_bounded(tmp_path):
    command = _mini_command(
        "mini", prompt="assignment", config_path=tmp_path / "mini.yaml",
        trajectory_path=tmp_path / "runtime" / "trajectory.json")
    config = _mini_config(
        tmp_path / "worktree", max_steps=17, timeout_seconds=91,
        requested_seed=123, forward_seed=True, temperature=0.2)

    assert command[:3] == ["mini", "-c", "mini.yaml"]
    assert command[-2:] == ["-o", str(tmp_path / "runtime" / "trajectory.json")]
    assert config["agent"]["step_limit"] == 17
    assert config["agent"]["wall_time_limit_seconds"] == 91
    assert config["environment"]["cwd"] == str(tmp_path / "worktree")
    assert config["model"]["model_kwargs"]["seed"] == 123
    assert config["model"]["model_kwargs"]["temperature"] == 0.2
    assert config["model"]["cost_tracking"] == "ignore_errors"


def test_agy_uses_hlink_native_auth_and_an_ephemeral_worktree_project(tmp_path):
    command = _agy_command(
        "hlink", worktree=tmp_path / "worktree",
        model="gemini-3.7-flash-low", timeout_seconds=91,
        log_path=tmp_path / "runtime" / "agy.log")

    assert command[:2] == ["hlink", "agy"]
    assert command[command.index("-p") + 1] == "-"
    assert "--new-project" in command
    assert command[command.index("--add-dir") + 1] == str(tmp_path / "worktree")
    assert command[command.index("--print-timeout") + 1] == "91s"
    assert "--dangerously-skip-permissions" not in command
    assert "-y" in command


def test_agy_rejects_provider_adapter_because_it_uses_native_login():
    with pytest.raises(ValueError, match="native authenticated provider"):
        _resolve_harness_executable("agy", "harness-link", "agy")


def test_direct_mini_is_rejected_until_provider_config_is_supported():
    with pytest.raises(ValueError, match="requires --adapter harness-link"):
        _resolve_harness_executable("mini", "direct", "mini")


def test_mini_version_comes_from_its_own_environment(tmp_path, monkeypatch):
    executable = tmp_path / "bin" / "mini"
    executable.parent.mkdir()
    executable.touch()
    observed = {}

    def check_output(command, text):
        observed["command"] = command
        return "1.15.0\n"

    monkeypatch.setattr("subprocess.check_output", check_output)

    assert _harness_version("mini", str(executable)) == "1.15.0"
    assert observed["command"][0] == str(executable.parent / "python")


def test_harness_link_probes_the_opencode_it_resolves_from_path(monkeypatch):
    resolved = {
        "opencode": "/actual/path/opencode",
        "/unused/custom-opencode": "/unused/custom-opencode",
    }
    monkeypatch.setattr("shutil.which", resolved.get)

    executable = _resolve_opencode_executable(
        "harness-link", "/unused/custom-opencode")

    assert executable == "/actual/path/opencode"


def test_systemd_resource_wrapper_records_hard_limits():
    command = _resource_command(["bwrap", "true"], {
        "enabled": True,
        "executable": "/usr/bin/systemd-run",
        "memory_max": "8G",
        "tasks_max": 512,
        "cpu_quota": "400%",
    })

    assert "MemoryMax=8G" in command
    assert "TasksMax=512" in command
    assert "CPUQuota=400%" in command
    assert command[-2:] == ["bwrap", "true"]


def test_scope_check_rejects_sibling_paths():
    owned = "reasoning_core/tasks/mutated/wave0/n01"
    paths = [f"{owned}/task.py", "reasoning_core/tasks/regex.py"]

    assert _outside_owned(paths, owned) == ["reasoning_core/tasks/regex.py"]


def test_queue_and_explicit_trials_are_combined_in_plan_order():
    plan = load_plan(PLAN)

    selected = _select_trials(plan, ("N2",), ("pilot",))

    assert [trial.trial_id for trial in selected] == ["N2", "N4", "M1"]


def test_sample_review_hard_gate_uses_durable_artifacts(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    sample = root / "samples_N1.md"
    (root / "generate_samples_N1.py").write_text("# sample generator\n")
    sample.write_text(
        "# Level 0\nPrompt: a\nAnswer: b\nPrompt: c\nAnswer: d\n"
        "# Level 2\nPrompt: e\nAnswer: f\nPrompt: g\nAnswer: h\n"
        "# Level 5\nPrompt: i\nAnswer: j\nPrompt: k\nAnswer: l\n"
    )
    events = tmp_path / "events.jsonl"
    target = str(sample)
    events.write_text("\n".join([
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "bash", "state": {
                "status": "completed", "input": {
                    "command": "cd /repo && PYTHONDONTWRITEBYTECODE=1 python "
                    f"{owned}/generate_samples_N1.py"},
                "metadata": {"exit": 0}}},
        }),
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "write", "state": {
                "status": "completed", "input": {"filePath": target}}},
        }),
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "read", "state": {
                "status": "completed", "input": {"filePath": target}}},
        }),
    ]))

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["ok"] is True


def test_sample_command_exit_is_observed_but_not_a_hard_gate(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text(
        "# Level 0\nPrompt: a\nAnswer: b\nPrompt: c\nAnswer: d\n"
        "# Level 2\nPrompt: e\nAnswer: f\nPrompt: g\nAnswer: h\n"
        "# Level 5\nPrompt: i\nAnswer: j\nPrompt: k\nAnswer: l\n"
    )
    command = (
        "PYTHONDONTWRITEBYTECODE=1 python "
        f"{owned}/generate_samples_N1.py"
    )
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "bash", "state": {
                "status": "completed", "input": {"command": command},
                "metadata": {"exit": 1}}},
        }),
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "read", "state": {
                "status": "completed", "input": {"filePath": str(sample)}}},
        }),
    ]))

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["command_succeeded"] is False
    assert review["ok"] is True


def test_sample_event_order_is_observational_not_a_hard_gate(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text(
        "# Level 0\nPrompt: a\nAnswer: b\nPrompt: c\nAnswer: d\n"
        "# Level 2\nPrompt: e\nAnswer: f\nPrompt: g\nAnswer: h\n"
        "# Level 5\nPrompt: i\nAnswer: j\nPrompt: k\nAnswer: l\n"
    )
    # Built, not spelled out: the recorded command has to match the one the harness
    # runs, and it has gained a PYTHONPATH since this test was written.
    command = _sample_command_for(owned, "N1")
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "read", "state": {
                "status": "completed", "input": {"filePath": str(sample)}}},
        }),
        json.dumps({
            "type": "tool_use",
            "part": {"tool": "bash", "state": {
                "status": "completed", "input": {"command": command},
                "metadata": {"exit": 0}}},
        }),
    ]))

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["command_succeeded"] is True
    assert review["read_after_last_edit"] is False
    assert review["ok"] is True


def test_sample_review_observes_agy_tools(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text(
        "# Level 0\nPrompt: a\nAnswer: b\nPrompt: c\nAnswer: d\n"
        "# Level 2\nPrompt: e\nAnswer: f\nPrompt: g\nAnswer: h\n"
        "# Level 5\nPrompt: i\nAnswer: j\nPrompt: k\nAnswer: l\n"
    )
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({"event": "step_update", "step_update": {
            "step_type": "tool", "state": "DONE", "tool_name": "run_command",
            "tool_info": {"parameters": {
                "CommandLine": _sample_command_for(owned, "N1")}}}}),
        json.dumps({"event": "step_update", "step_update": {
            "step_type": "tool", "state": "DONE", "tool_name": "view_file",
            "tool_info": {"parameters": {"AbsolutePath": str(sample)}}}}),
    ]))

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["command_succeeded"] is True
    assert review["read_after_last_edit"] is True
    assert review["ok"] is True


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

    assert results == [{
        "command": "/bin/sleep 2",
        "exit_code": 124,
        "timed_out": True,
    }]


def test_validation_environment_removes_named_credentials(monkeypatch):
    monkeypatch.setenv("ALBERT_API_KEY", "secret")
    monkeypatch.setenv("KEEP_ME", "visible")

    environment = _sanitized_environment(("ALBERT_API_KEY",))

    assert "ALBERT_API_KEY" not in environment
    assert environment["KEEP_ME"] == "visible"


def test_sandboxed_validation_does_not_receive_named_credential(monkeypatch):
    monkeypatch.setenv("ALBERT_API_KEY", "secret")
    with tempfile.TemporaryDirectory(prefix=".task-search-test-", dir=ROOT) as root:
        root = Path(root)
        worktree = root / "worktree"
        (worktree / "owned").mkdir(parents=True)
        results = _run_validation(
            worktree,
            ('test -z "${ALBERT_API_KEY+x}"',),
            root / "validation.log",
            owned_path="owned",
            runtime_root=root / "runtime",
            bwrap_bin="bwrap",
            resource_limits={"enabled": False},
            timeout_seconds=5,
            credential_env_names=("ALBERT_API_KEY",),
        )

    assert results[0]["exit_code"] == 0


def test_sample_generator_command_is_allowed():
    trial = load_plan(PLAN).trials[0]
    permissions = opencode_config(trial, "task-search-worker")["permission"]

    assert permissions["bash"][_sample_command(trial)] == "allow"


def test_task_classes_ignores_tests_and_finds_owned_task(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "task.py").write_text("class Example(Task):\n    pass\n")
    (root / "test_task.py").write_text("class FakeTask(Task):\n    pass\n")

    assert _task_classes(tmp_path, owned) == [
        ("reasoning_core.tasks.generated.wave.example.task", "Example")
    ]


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


def test_self_check_is_the_only_verification_command_the_prompt_asks_for(tmp_path):
    """The prompt hands out one verification command and the sandbox allows exactly it.

    Trials were spending half a 28-step budget on five separate verification commands,
    and the gates that were not among them -- TASK_META, the contract audit -- only
    surfaced in run.json once the trial was already lost.
    """
    trial = Trial(
        trial_id="N1", instruction="Implement it.",
        owned_path="reasoning_core/tasks/generated/wave/example",
        validation=("PYTHONDONTWRITEBYTECODE=1 python -m pytest reasoning_core/tasks",),
        hypothesis="N1")
    command = _selfcheck_command_for(trial.owned_path, trial.trial_id)
    assert command in opencode_permissions(trial)["bash"]
    assert opencode_permissions(trial)["bash"][command] == "allow"

    plan = SearchPlan(name="wave", base_ref="HEAD", context_files=(), trials=(trial,),
                      queues={}, path=tmp_path / "plan.yaml")
    plan.path.write_text("version: 1\n")
    prompt = render_prompt(plan, trial, Path("."))
    assert command in prompt
    # The recipes it replaced are gone: no hand-rolled reproducibility check, no
    # separately quoted prior_audit invocation.
    assert "PYTHONHASHSEED" not in prompt
    assert "prior_audit" not in prompt


def test_selfcheck_pytest_stops_at_an_actionable_short_traceback():
    command = selfcheck.pytest_command("owned/task")

    assert "-q -x --tb=short" in command
    assert command.endswith("--import-mode=importlib owned/task")


def test_live_trajectory_reads_budget_from_invocation_summary(tmp_path):
    wave = tmp_path / "run"
    trial = wave / "S26"
    trial.mkdir(parents=True)
    (wave / "summary.json").write_text(json.dumps({"max_steps": 28}))
    (trial / "events.jsonl").write_text(
        json.dumps({"type": "step_start", "part": {}}) + "\n")

    row = trajectory.read(trial)

    assert row["status"] == "no run.json"
    assert row["steps"] == 1
    assert row["budget"] == 28


def test_trajectory_marks_a_truncated_selfcheck_incomplete(tmp_path):
    trial = tmp_path / "run" / "S35"
    trial.mkdir(parents=True)
    event = {"type": "tool_use", "part": {"tool": "bash", "state": {
        "status": "completed",
        "input": {"command": "python -m reasoning_core.task_search.selfcheck owned S35"},
        "output": "implementation PASS\nspeed PASS\n<shell_metadata>timed out</shell_metadata>",
    }}}
    (trial / "events.jsonl").write_text(json.dumps(event) + "\n")

    row = trajectory.read(trial)

    assert row["checks"][-1]["incomplete"] == "FAIL"


def test_trajectory_reads_agy_stream_json(tmp_path):
    trial = tmp_path / "run" / "S41"
    trial.mkdir(parents=True)
    events = [
        {"event": "step_update", "step_update": {
            "step_type": "agent_response", "state": "DONE"}},
        {"event": "step_update", "step_update": {
            "step_type": "tool", "state": "DONE", "tool_name": "run_command",
            "tool_info": {"parameters": {"CommandLine":
                "python -m reasoning_core.task_search.selfcheck owned S41"},
                "output": "implementation PASS\n0 gate(s) failing.\n"}}},
        {"event": "result", "result": {"status": "SUCCESS"}},
    ]
    (trial / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events))

    row = trajectory.read(trial)

    assert row["steps"] == 1
    assert row["stopped"] == "SUCCESS"
    assert row["checks"] == [{"implementation": "PASS"}]


def test_selfcheck_distinguishes_probe_crashes_from_slowness():
    crashed = selfcheck.speed_failure(1, "Traceback\nAssertionError: broken")
    timed_out = selfcheck.speed_failure(124, "killed")

    assert "probe crashed" in crashed
    assert "AssertionError: broken" in crashed
    assert "did not finish" in timed_out


def test_owned_digest_sees_a_file_rewritten_after_the_contract_audit(tmp_path):
    """The freeze gate: model-authored tests run with the owned directory writable."""
    owned = tmp_path / "reasoning_core" / "tasks" / "generated" / "n1"
    owned.mkdir(parents=True)
    (owned / "task.py").write_text("GOLD = 1\n")
    (owned / "samples_N1.md").write_text("level 0\n")
    relative = "reasoning_core/tasks/generated/n1"
    frozen = _owned_digest(tmp_path, relative, exclude=("samples_N1.md",))

    # The generator rewriting its own output is allowed and must not trip the gate.
    (owned / "samples_N1.md").write_text("level 0 answer answer level 2 answer"
                                        " answer level 5 answer answer\n")
    assert _owned_digest(tmp_path, relative, exclude=("samples_N1.md",)) == frozen

    (owned / "task.py").write_text("GOLD = 2\n")
    after = _owned_digest(tmp_path, relative, exclude=("samples_N1.md",))
    assert after["files"]["task.py"] != frozen["files"]["task.py"]
    assert after["tree_sha256"] != frozen["tree_sha256"]


def test_owned_digest_sees_a_swap_that_leaves_the_bytes_alone(tmp_path):
    """Content is not the file. A digest over bytes alone certifies the wrong thing.

    Both swaps here read back identical content through the path that was frozen, so a
    content-only hash calls the candidate unchanged: one hands the accepted task.py to
    a file outside the owned directory, the other only flips a mode bit.
    """
    owned = tmp_path / "reasoning_core" / "tasks" / "generated" / "n1"
    owned.mkdir(parents=True)
    (owned / "task.py").write_text("GOLD = 1\n")
    relative = "reasoning_core/tasks/generated/n1"
    frozen = _owned_digest(tmp_path, relative)

    (tmp_path / "elsewhere.py").write_text("GOLD = 1\n")
    (owned / "task.py").unlink()
    (owned / "task.py").symlink_to(tmp_path / "elsewhere.py")
    assert (owned / "task.py").read_text() == "GOLD = 1\n"
    swapped = _owned_digest(tmp_path, relative)
    assert swapped["tree_sha256"] != frozen["tree_sha256"]

    (owned / "task.py").unlink()
    (owned / "task.py").write_text("GOLD = 1\n")
    assert _owned_digest(tmp_path, relative) == frozen
    (owned / "task.py").chmod(0o755)
    assert _owned_digest(tmp_path, relative)["tree_sha256"] != frozen["tree_sha256"]


def test_self_check_reports_the_sections_the_gate_actually_requires():
    """The self-check is the worker's only view of this gate, so its copy has to match.

    selfcheck runs inside the sandbox and cannot import the runner to share the code,
    so the two copies are pinned here by behaviour -- a worker told it passed a gate the
    coordinator then fails it on is the failure this whole harness exists to avoid.
    """
    thorough = ("## Level 0\nP\nAnswer: 1\nP\nAnswer: 2\n"
                "## Level 2\nP\nAnswer: 3\nP\nAnswer: 4\n"
                "## Level 5\nP\nAnswer: 5\nP\nAnswer: 6\n")
    # One example per level: the headings are all there, which is all the gate used
    # to look for. Three of 480 sample files looked exactly like this.
    thin = thorough.replace("P\nAnswer: 2\n", "").replace("P\nAnswer: 4\n", "")
    for body in (thorough, thin, "", "Level 0\nLevel 2\nLevel 5\nAnswer"):
        assert sample_shortfall(body) == selfcheck.sample_shortfall(body)
    assert sample_shortfall(thorough) == []
    assert [s.split()[1] for s in sample_shortfall(thin)] == ["0", "2"]
    assert len(sample_shortfall("")) == 3


def test_plan_problems_are_the_ones_check_used_to_miss():
    """A plan could pass `check` and still have nowhere to run.

    Each of these surfaced only at launch, after the worktrees were made: an owned path
    the contract audit cannot turn into an import, a context file render_prompt would
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
        "base_ref does not resolve to a commit: no-such-ref"]


def test_frozen_module_drift_catches_a_base_ref_left_behind(tmp_path):
    """Workers run the harness modules frozen at base_ref; the gates are whatever is live.

    Nothing else in the harness compares those two, so a gate tightened without moving
    base_ref forward would go out silently -- and the worker it fails would have been
    told, by the harness itself, that it had passed. A flag added to prior_audit is
    worse still: the coordinator writes the command line live, so the pinned copy is
    handed an argument it has never heard of.
    """
    paths = {name: tmp_path / f"reasoning_core/task_search/{name}.py"
             for name in ("selfcheck", "prior_audit")}
    git = lambda *args: subprocess.run(("git",) + args, cwd=tmp_path, check=True,
                                       capture_output=True)
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


def test_prior_audit_reports_a_level_the_generator_cannot_reach():
    """The speed gate times the default config, so a dead top level passed unseen.

    A generated subset-optimisation task passed all eleven gates and could not produce
    a single level 6 example: its search blew template's per-example timeout, which is
    raised from a signal handler and does not descend from Exception.
    """
    class DiesAtDepth:
        config = types.SimpleNamespace(set_level=lambda level: None)

        def generate_example(self):
            raise selfcheck_timeout()

        def score_answer(self, answer, entry):
            return 0.0

    report = prior_audit.audit(DiesAtDepth(), 6, 20, time.time() + 20)
    assert report["n"] == 0 and "Boom" in report["error"]


class selfcheck_timeout(BaseException):
    """Stands in for template.TimeoutException, which is not an Exception either."""
    def __str__(self):
        return "Boom"


def test_self_check_and_coordinator_audit_on_the_same_thresholds():
    """Two files spell out this command line; a gate added to one only is a lie.

    The self-check reports `gameability` to the worker. If the coordinator audits at a
    threshold the self-check does not pass, the worker is told it passed a gate the
    harness is about to fail it on.
    """
    command = _prior_audit_command(load_plan(PLAN).trials[0])
    source = (ROOT / "reasoning_core/task_search/selfcheck.py").read_text()
    for flag in ("--max-const", "--max-shortcut"):
        value = command.split(flag)[1].split()[0]
        assert f"{flag} {value}" in source, f"{flag} differs between selfcheck and runner"


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
            return types.SimpleNamespace(prompt=f"the total is {n}. What is the total?",
                                         answer=str(n))
        def score_answer(self, answer, entry):
            return float(str(answer) == entry.answer)

    task = Copyable()
    report = prior_audit.audit(task, 0, 20, time.time() + 20)
    assert report["const"] < 0.4 and report["distinct"] > 0.9
    assert report["shortcut"] == 1.0 and report["rule"] == "last_number"


def test_undiscoverable_flags_what_the_audit_imports_but_discovery_skips():
    hidden = _undiscoverable([
        ("reasoning_core.tasks.generated.wave0.n04_x.task", "T"),
        ("reasoning_core.tasks.generated.wave0._hidden.task", "T"),
        ("reasoning_core.tasks.generated.wave0.n04_x._task", "T"),
        ("reasoning_core.tasks.deprecated.old.task", "T"),
    ])
    assert hidden == [
        "reasoning_core.tasks.deprecated.old.task",
        "reasoning_core.tasks.generated.wave0._hidden.task",
        "reasoning_core.tasks.generated.wave0.n04_x._task",
    ]


@pytest.mark.parametrize("source, error", [
    ("TASK_META = {\n", SyntaxError),
    ("TASK_META = dict(idea='x')\n", ValueError),
])
def test_unparseable_candidate_metadata_raises_something_catchable(tmp_path, source, error):
    """It used to reach run_plan and be recorded as orchestration_error, a runner bug."""
    owned = tmp_path / "reasoning_core" / "tasks" / "generated" / "n1"
    owned.mkdir(parents=True)
    (owned / "task.py").write_text(source)
    with pytest.raises(error):
        _task_metadata(tmp_path, "reasoning_core/tasks/generated/n1")


def test_pace_changes_the_prompt_and_nothing_else():
    """The hurry stance is an assumption about the bottleneck, so it has to be A/B-able.

    Only the two pacing strings may differ between arms. Substituting them out has to
    leave three byte-identical prompts, or the arm is confounded with whatever else
    moved -- which is how `pace` leaking into TASK_META was caught.
    """
    plan = load_plan(PLAN)
    trial = plan.trials[0]
    prompts = {name: render_prompt(plan, trial, ROOT, pace=name) for name in PACE}

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


def test_step_usage_flags_a_worker_that_ran_out_of_budget(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text("".join(
        json.dumps({"type": "step_start"}) + "\n" for _ in range(28)))
    assert _step_usage(events, 28) == {"used": 28, "max": 28, "exhausted": True}
    assert _step_usage(events, 60)["exhausted"] is False
    assert _step_usage(tmp_path / "absent.jsonl", 28) is None


def test_step_usage_understands_agy_stream_json(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({"event": "step_update", "step_update": {
            "step_type": "agent_response", "state": "DONE"}}),
        json.dumps({"event": "step_update", "step_update": {
            "step_type": "tool", "state": "DONE"}}),
        json.dumps({"event": "step_update", "step_update": {
            "step_type": "agent_response", "state": "DONE"}}),
    ]))

    assert _step_usage(events, None) == {
        "used": 2, "max": None, "exhausted": False}


def test_sample_sanity_fails_open_without_a_reviewer(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_SEARCH_REVIEW_KEY_ENV", "TASK_SEARCH_ABSENT_KEY")
    monkeypatch.delenv("TASK_SEARCH_ABSENT_KEY", raising=False)
    assert _sample_sanity(tmp_path / "samples_S1.md") == {
        "verdict": None, "why": "no reviewer key"}


def test_sample_sanity_reads_the_verdict_and_reason(tmp_path, monkeypatch):
    import io, json as _json
    samples = tmp_path / "samples_S1.md"
    samples.write_text("Answer: -44/5\n")
    monkeypatch.setenv("TASK_SEARCH_REVIEW_KEY_ENV", "TASK_SEARCH_FAKE_KEY")
    monkeypatch.setenv("TASK_SEARCH_FAKE_KEY", "x")
    reply = _json.dumps({"choices": [{"message": {"content":
        "VERDICT: INVALID\nWHY: expected time is negative."}}]})
    requests = []
    def urlopen(request, **kwargs):
        requests.append(_json.loads(request.data))
        return io.BytesIO(reply.encode())
    monkeypatch.setattr("reasoning_core.task_search.runner.urllib.request.urlopen", urlopen)
    assert _sample_sanity(samples, instruction="counts stay non-negative",
                          source="answer = -44 / 5") == {
        "verdict": "INVALID", "why": "expected time is negative."}
    review_text = requests[0]["messages"][1]["content"]
    assert "counts stay non-negative" in review_text
    assert "answer = -44 / 5" in review_text


def test_sample_sanity_fails_open_on_empty_model_content(tmp_path, monkeypatch):
    import io
    samples = tmp_path / "samples_S1.md"
    samples.write_text("Answer: 1\n")
    monkeypatch.setenv("TASK_SEARCH_REVIEW_KEY_ENV", "TASK_SEARCH_FAKE_KEY")
    monkeypatch.setenv("TASK_SEARCH_FAKE_KEY", "x")
    reply = json.dumps({"choices": [{"message": {"content": None}}]})
    monkeypatch.setattr("reasoning_core.task_search.runner.urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(reply.encode()))

    assert _sample_sanity(samples) == {
        "verdict": None, "why": "reviewer returned no text"}


def test_review_source_excludes_tests_and_sample_generators(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "task.py").write_text("ANSWER = -1\n")
    (root / "_draft.py").write_text("BROKEN = True\n")
    (root / "test_task.py").write_text("assert True\n")
    (root / "generate_samples_X.py").write_text("print('samples')\n")

    source = _review_source(tmp_path, owned)

    assert "ANSWER = -1" in source
    assert "BROKEN = True" not in source
    assert "assert True" not in source
    assert "print('samples')" not in source
