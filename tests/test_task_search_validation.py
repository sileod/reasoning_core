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
    FAILURE_PRECEDENCE,
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
    classify,
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


def test_failure_precedence_is_encoded_once():
    checks = {name: True for _, name in FAILURE_PRECEDENCE}
    checks["contract"] = False
    checks["semantics"] = False

    assert classify(checks) == "contract_failed"

def test_scope_check_rejects_sibling_paths():
    owned = "reasoning_core/tasks/mutated/wave0/n01"
    paths = [f"{owned}/task.py", "reasoning_core/tasks/regex.py"]

    assert _outside_owned(paths, owned) == ["reasoning_core/tasks/regex.py"]

def test_sample_review_hard_gate_uses_durable_artifacts(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    sample = root / "samples_N1.md"
    (root / "generate_samples_N1.py").write_text("# sample generator\n")
    sample.write_text(SAMPLE_BODY)
    events = tmp_path / "events.jsonl"
    target = str(sample)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {
                                    "command": "cd /repo && PYTHONDONTWRITEBYTECODE=1 python "
                                    f"{owned}/generate_samples_N1.py"
                                },
                                "metadata": {"exit": 0},
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "write",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": target},
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": target},
                            },
                        },
                    }
                ),
            ]
        )
    )

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["ok"] is True

def test_sample_command_exit_is_observed_but_not_a_hard_gate(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text(SAMPLE_BODY)
    command = "PYTHONDONTWRITEBYTECODE=1 python " f"{owned}/generate_samples_N1.py"
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": command},
                                "metadata": {"exit": 1},
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": str(sample)},
                            },
                        },
                    }
                ),
            ]
        )
    )

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["command_succeeded"] is False
    assert review["ok"] is True

def test_sample_event_order_is_observational_not_a_hard_gate(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "generate_samples_N1.py").write_text("# generator\n")
    sample = root / "samples_N1.md"
    sample.write_text(SAMPLE_BODY)
    # Built, not spelled out: the recorded command has to match the one the harness
    # runs, and it has gained a PYTHONPATH since this test was written.
    command = _sample_command_for(owned, "N1")
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": str(sample)},
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": command},
                                "metadata": {"exit": 0},
                            },
                        },
                    }
                ),
            ]
        )
    )

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
    sample.write_text(SAMPLE_BODY)
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_type": "tool",
                            "state": "DONE",
                            "tool_name": "run_command",
                            "tool_info": {
                                "parameters": {
                                    "CommandLine": _sample_command_for(owned, "N1")
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_type": "tool",
                            "state": "DONE",
                            "tool_name": "view_file",
                            "tool_info": {"parameters": {"AbsolutePath": str(sample)}},
                        },
                    }
                ),
            ]
        )
    )

    review = _sample_review(tmp_path, owned, "N1", events)

    assert review["command_succeeded"] is True
    assert review["read_after_last_edit"] is True
    assert review["ok"] is True

def test_task_classes_ignores_tests_and_finds_owned_task(tmp_path):
    owned = "reasoning_core/tasks/generated/wave/example"
    root = tmp_path / owned
    root.mkdir(parents=True)
    (root / "task.py").write_text("class Example(Task):\n    pass\n")
    (root / "test_task.py").write_text("class FakeTask(Task):\n    pass\n")

    assert _task_classes(tmp_path, owned) == [
        ("reasoning_core.tasks.generated.wave.example.task", "Example")
    ]

def test_selfcheck_pytest_stops_at_an_actionable_short_traceback():
    command = validation.pytest_command("owned/task")

    assert "-q -x --tb=short" in command
    assert command.endswith("--import-mode=importlib owned/task")

def test_selfcheck_distinguishes_probe_crashes_from_slowness():
    crashed = validation.speed_failure(1, "Traceback\nAssertionError: broken")
    timed_out = validation.speed_failure(124, "killed")

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
    (owned / "samples_N1.md").write_text(
        "level 0 answer answer level 2 answer" " answer level 5 answer answer\n"
    )
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
    """The shared samples gate rejects incomplete and prompt-free examples."""
    thorough = SAMPLE_BODY.replace("# Level", "## Level")
    # One example per level: the headings are all there, which is all the gate used
    # to look for. Three of 480 sample files looked exactly like this.
    thin = thorough.replace(SAMPLE_PROMPT + "Answer: 00\n", "", 1).replace(
        SAMPLE_PROMPT + "Answer: 21\n", "", 1
    )
    # Every heading and every answer, no prompt under any of them: S45 in wave4.
    empty = "".join(
        f"## Level {level}\nAnswer: 1\nAnswer: 2\n" for level in ("0", "2", "5")
    )
    assert sample_shortfall(thorough) == []
    assert [s.split()[1] for s in sample_shortfall(thin)] == ["0", "2"]
    assert [s.split()[1] for s in sample_shortfall(empty)] == ["0", "2", "5"]
    assert all("prompt text" in s for s in sample_shortfall(empty))
    assert len(sample_shortfall("")) == 6

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

def test_undiscoverable_flags_what_the_audit_imports_but_discovery_skips():
    hidden = _undiscoverable(
        [
            ("reasoning_core.tasks.generated.wave0.n04_x.task", "T"),
            ("reasoning_core.tasks.generated.wave0._hidden.task", "T"),
            ("reasoning_core.tasks.generated.wave0.n04_x._task", "T"),
            ("reasoning_core.tasks.deprecated.old.task", "T"),
        ]
    )
    assert hidden == [
        "reasoning_core.tasks.deprecated.old.task",
        "reasoning_core.tasks.generated.wave0._hidden.task",
        "reasoning_core.tasks.generated.wave0.n04_x._task",
    ]

@pytest.mark.parametrize(
    "source, error",
    [
        ("TASK_META = {\n", SyntaxError),
        ("TASK_META = dict(idea='x')\n", ValueError),
    ],
)
def test_unparseable_candidate_metadata_raises_something_catchable(
    tmp_path, source, error
):
    """It used to reach run_plan and be recorded as orchestration_error, a runner bug."""
    owned = tmp_path / "reasoning_core" / "tasks" / "generated" / "n1"
    owned.mkdir(parents=True)
    (owned / "task.py").write_text(source)
    with pytest.raises(error):
        _task_metadata(tmp_path, "reasoning_core/tasks/generated/n1")

def test_sample_sanity_fails_open_without_a_reviewer(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_SEARCH_REVIEW_KEY_ENV", "TASK_SEARCH_ABSENT_KEY")
    monkeypatch.delenv("TASK_SEARCH_ABSENT_KEY", raising=False)
    assert _sample_sanity(tmp_path / "samples_S1.md") == {
        "verdict": None,
        "why": "no reviewer key",
    }

def test_sample_sanity_reads_the_verdict_and_reason(tmp_path, monkeypatch):
    import io, json as _json

    samples = tmp_path / "samples_S1.md"
    samples.write_text("Answer: -44/5\n")
    monkeypatch.setenv("TASK_SEARCH_REVIEW_KEY_ENV", "TASK_SEARCH_FAKE_KEY")
    monkeypatch.setenv("TASK_SEARCH_FAKE_KEY", "x")
    monkeypatch.setenv(
        "TASK_SEARCH_REVIEW_ENDPOINT", "https://example.test/v1/chat/completions"
    )
    monkeypatch.setenv("TASK_SEARCH_REVIEW_MODEL", "example-model")
    reply = _json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "VERDICT: INVALID\nWHY: expected time is negative."
                    }
                }
            ]
        }
    )
    requests = []

    def urlopen(request, **kwargs):
        requests.append(_json.loads(request.data))
        return io.BytesIO(reply.encode())

    monkeypatch.setattr(
        "reasoning_core.task_search.validation.urllib.request.urlopen", urlopen
    )
    assert _sample_sanity(
        samples, instruction="counts stay non-negative", source="answer = -44 / 5"
    ) == {"verdict": "INVALID", "why": "expected time is negative."}
    review_text = requests[0]["messages"][1]["content"]
    assert "counts stay non-negative" in review_text
    assert "answer = -44 / 5" in review_text

def test_sample_sanity_fails_open_on_empty_model_content(tmp_path, monkeypatch):
    import io

    samples = tmp_path / "samples_S1.md"
    samples.write_text("Answer: 1\n")
    monkeypatch.setenv("TASK_SEARCH_REVIEW_KEY_ENV", "TASK_SEARCH_FAKE_KEY")
    monkeypatch.setenv("TASK_SEARCH_FAKE_KEY", "x")
    monkeypatch.setenv(
        "TASK_SEARCH_REVIEW_ENDPOINT", "https://example.test/v1/chat/completions"
    )
    monkeypatch.setenv("TASK_SEARCH_REVIEW_MODEL", "example-model")
    reply = json.dumps({"choices": [{"message": {"content": None}}]})
    monkeypatch.setattr(
        "reasoning_core.task_search.validation.urllib.request.urlopen",
        lambda *a, **k: io.BytesIO(reply.encode()),
    )

    assert _sample_sanity(samples) == {
        "verdict": None,
        "why": "reviewer returned no text",
    }

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
