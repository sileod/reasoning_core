import json
from pathlib import Path
import tempfile

import pytest

from reasoning_core.task_search.runner import (
    _adapter_command,
    _outside_owned,
    _opencode_command,
    _resource_command,
    _run_validation,
    _sandbox_command,
    _sample_review,
    _sample_command,
    _select_trials,
    _task_classes,
    generation_metadata,
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


def test_sample_review_requires_sections_and_read_after_write(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    sample = root / "samples_N1.md"
    (root / "generate_samples_N1.py").write_text("# sample generator\n")
    sample.write_text(
        "# Level 0\nPrompt: a\nAnswer: b\n"
        "# Level 2\nPrompt: c\nAnswer: d\n"
        "# Level 5\nPrompt: e\nAnswer: f\n"
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


def test_sample_review_uses_shell_exit_code_not_tool_completion(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text("# Level 0\nAnswer\n# Level 2\nAnswer\n# Level 5\nAnswer\n")
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
    assert review["ok"] is False


def test_sample_review_requires_read_after_last_generator_run(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text("# Level 0\nAnswer\n# Level 2\nAnswer\n# Level 5\nAnswer\n")
    command = (
        "PYTHONDONTWRITEBYTECODE=1 python "
        f"{owned}/generate_samples_N1.py"
    )
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
    assert review["ok"] is False


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
