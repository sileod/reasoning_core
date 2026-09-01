"""The worker permission spellings, and the trajectory digest that needs no network."""

import json

from reasoning_core.task_search import digest
from reasoning_core.task_search.plan import Trial
from reasoning_core.task_search.runner import _spellings, opencode_permissions

OWNED = "reasoning_core/tasks/generated/wave7/graph_chordality"


def _trial():
    return Trial(
        trial_id="T008",
        instruction="author one task",
        owned_path=OWNED,
        validation=(f"PYTHONDONTWRITEBYTECODE=1 python -m pytest {OWNED}",),
        idea="idea",
        changes="changes",
    )


def test_a_prescribed_command_is_allowed_however_the_worker_spells_it():
    bash = opencode_permissions(_trial())["bash"]
    # Each of these was denied in wave7, and each is the audit the harness told the
    # worker to run. T008 stopped believing it was finished with the audit unrun.
    for command in (
        f"python -m reasoning_core.task_search.prior_audit --path {OWNED} --n 30",
        f"PYTHONPATH=. python -m reasoning_core.task_search.prior_audit --path {OWNED}",
        f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m"
        f" reasoning_core.task_search.prior_audit --path {OWNED} --max-const 0.4",
    ):
        assert bash.get(command) == "allow" or any(
            pattern.endswith("*") and command.startswith(pattern[:-1])
            for pattern, action in bash.items()
            if action == "allow"
        ), command


def test_the_wildcard_does_not_widen_a_command_to_another_trials_path():
    patterns = _spellings(
        f"python -m reasoning_core.task_search.prior_audit --path {OWNED} --n 30", OWNED
    )
    assert not any(
        pattern[:-1] and "reasoning_core/tasks/generated/OTHER".startswith(pattern[:-1])
        for pattern in patterns
    )
    assert all(OWNED in pattern for pattern in patterns if pattern.endswith("*"))


def test_a_command_without_the_owned_path_gets_no_truncated_wildcard():
    patterns = _spellings("git status", OWNED)
    assert set(patterns) == {
        prefix + head
        for prefix in ("", "PYTHONDONTWRITEBYTECODE=1 ", "PYTHONPATH=. ",
                       "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ")
        for head in ("git status", "git status*")
    }


def test_the_digest_asks_about_failures_and_near_misses_only():
    rows = [
        {"id": "A", "status": "success", "steps": 9, "budget": 40},
        {"id": "B", "status": "validation_failed", "steps": 12, "budget": 40},
        {"id": "C", "status": "success", "steps": 40, "budget": 40},
    ]
    assert [row["id"] for row in digest.interesting(rows)] == ["B", "C"]


def test_a_wave_that_worked_costs_no_model_call(tmp_path, capsys, monkeypatch):
    trial = tmp_path / "T001"
    trial.mkdir()
    (trial / "events.jsonl").write_text("")
    (trial / "run.json").write_text(json.dumps({"status": "success", "steps": {"max": 40}}))

    def refuse(*args, **kwargs):
        raise AssertionError("a clean wave must not spend a call")

    monkeypatch.setattr(digest, "ChatClient", refuse)
    assert digest.main([str(tmp_path)]) == 0
    assert "nothing to diagnose" in capsys.readouterr().out


def test_a_transcript_keeps_the_verdict_and_the_tail_not_the_whole_run(tmp_path):
    trial = tmp_path / "T009"
    trial.mkdir()
    (trial / "validation.log").write_text("E   AssertionError: sample 3 scored 0.0\n")
    row = {
        "id": "T009", "status": "validation_failed", "steps": 40, "budget": 40,
        "stopped": "step_limit", "checks": [{"pytest": "FAIL", "samples": "PASS"}],
        "denied": [], "errors": [], "summary": "I believe the task is complete.",
        "calls": [("bash", True, f"echo {index}", 10) for index in range(60)],
    }
    text = digest._transcript(trial, row)
    assert len(text) <= digest.TRIAL_CHARS
    assert "AssertionError: sample 3 scored 0.0" in text
    assert "echo 59" in text and "echo 0\n" not in text
    assert "pytest" in text
