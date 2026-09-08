from pathlib import Path

import pytest

from reasoning_core.task_search import cli


PLAN = Path(__file__).parents[2] / "reasoning_core/task_search/plans/wave0.yaml"


@pytest.mark.parametrize("flags, expected", [([], False), (["--snapshots"], True),
                                            (["--no-snapshots"], False)])
def test_run_forwards_snapshot_setting(monkeypatch, flags, expected):
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(cli, "run_plan", run)
    cli.main(["run", str(PLAN), "--model", "unused", *flags])
    assert seen["snapshots"] is expected


def test_check_returns_failure_for_plan_problems(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_plan_problems", lambda *args: ["missing parent module"])
    monkeypatch.setattr(cli, "_frozen_module_drift", lambda *args: None)
    with pytest.raises(SystemExit) as error:
        cli.main(["check", str(PLAN)])
    assert error.value.code == 1
    assert "PROBLEM: missing parent module" in capsys.readouterr().out


def test_run_prints_orchestration_error_reason(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_plan", lambda *args, **kwargs: [{
        "trial_id": "N1", "status": "orchestration_error", "error": "harness unavailable",
    }])
    with pytest.raises(SystemExit) as error:
        cli.main(["run", str(PLAN), "--model", "unused", "--trial", "N1"])
    assert error.value.code == 1
    assert "N1: harness unavailable" in capsys.readouterr().err
