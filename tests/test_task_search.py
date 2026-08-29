import json
from pathlib import Path
import tempfile

import pytest

from reasoning_core.task_search.runner import (
    _sample_command_for,
    opencode_permissions,
    Trial,
    SearchPlan,
    _selfcheck_command_for,
    _adapter_command,
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
    _undiscoverable,
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


def test_sample_command_exit_is_observed_but_not_a_hard_gate(tmp_path):
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
    assert review["ok"] is True


def test_sample_event_order_is_observational_not_a_hard_gate(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text("# Level 0\nAnswer\n# Level 2\nAnswer\n# Level 5\nAnswer\n")
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


def test_owned_digest_sees_a_file_rewritten_after_the_contract_audit(tmp_path):
    """The freeze gate: model-authored tests run with the owned directory writable."""
    owned = tmp_path / "reasoning_core" / "tasks" / "generated" / "n1"
    owned.mkdir(parents=True)
    (owned / "task.py").write_text("GOLD = 1\n")
    (owned / "samples_N1.md").write_text("level 0\n")
    relative = "reasoning_core/tasks/generated/n1"
    frozen = _owned_digest(tmp_path, relative, exclude=("samples_N1.md",))

    # The generator rewriting its own output is allowed and must not trip the gate.
    (owned / "samples_N1.md").write_text("level 0 level 2 level 5 answer\n")
    assert _owned_digest(tmp_path, relative, exclude=("samples_N1.md",)) == frozen

    (owned / "task.py").write_text("GOLD = 2\n")
    after = _owned_digest(tmp_path, relative, exclude=("samples_N1.md",))
    assert after["files"]["task.py"] != frozen["files"]["task.py"]
    assert after["tree_sha256"] != frozen["tree_sha256"]


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
