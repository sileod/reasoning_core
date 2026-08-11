import pytest

from reasoning_core.tasks import code_execution as code_tasks
from reasoning_core.tasks.code_execution import (
    CodeInputDeduction,
    CodeRunnability,
    MesopyCodeCfg,
    RunReport,
    endpoint_probes,
    function_triviality,
    organic_mutations,
    runnability_pair,
    sample_problem,
)


def report(value, args):
    return RunReport(ok=True, value=repr(value), args=args, steps=10)


def test_function_triviality():
    assert function_triviality([report(3, [1]), report(3, [2])]) == "constant"
    assert function_triviality([report(1, [1, 4]), report(2, [2, 5])]) == "identity"
    assert function_triviality([report(2, [1]), report(4, [2])]) is None


def test_sample_problem_rejects_trivial_function(monkeypatch):
    cfg = MesopyCodeCfg(
        max_attempts=1,
        min_steps=0,
        trivial_accept_prob=0,
        trivial_probes=2,
    )
    monkeypatch.setattr(code_tasks, "make_code", lambda *_: "def endpoint(x): return 1")
    monkeypatch.setattr(code_tasks, "run_code", lambda *_: report(1, [0]))

    with pytest.raises(RuntimeError):
        sample_problem(cfg, want_error=False, failure_rate=0)


def test_code_runnability_emits_paired_labels(monkeypatch):
    calls = []

    def paired(_):
        code = f"code-{len(calls)}"
        calls.append(code)
        return "name", ((code, RunReport(error="NameError", args=[0])), (code, report(4, [1])))

    monkeypatch.setattr(code_tasks, "runnability_pair", paired)
    task = CodeRunnability()
    problems = task.generate_balanced_batch(batch_size=4)

    assert len(calls) == 2
    for code in calls:
        pair = [problem for problem in problems if problem.metadata.code == code]
        assert {problem.answer for problem in pair} == {"OK", "NameError"}
    assert "The answer is `OK`" in task.prompt(problems[0].metadata)
    assert all(task.score_answer(problem.answer, problem) == 1.0 for problem in problems)
    assert not hasattr(task, "_pending_pair")


def test_runnability_pair_uses_declared_attempt_budget(monkeypatch):
    cfg = MesopyCodeCfg(max_attempts=4)
    calls = []
    monkeypatch.setattr(code_tasks, "make_code", lambda *_args, **_kwargs: calls.append(1) or "code")
    monkeypatch.setattr(code_tasks, "endpoint_probes", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError):
        runnability_pair(cfg)

    assert len(calls) == cfg.max_attempts


def test_code_generators_have_no_mutable_balancing_state():
    task = CodeInputDeduction()
    assert not hasattr(task, "_mode_i")
    assert not hasattr(task, "_recent_answers")
    assert task.balancing_key_ratio == pytest.approx(1 / 3)


def test_endpoint_probes_vary_each_annotated_argument():
    code = "def endpoint(x: int, s: str):\n    return x, s\n"
    probes = endpoint_probes(code, MesopyCodeCfg(), limit=24)

    assert len({x for x, _ in probes}) > 1
    assert len({s for _, s in probes}) > 1


def test_organic_mutations_are_local_edits_of_generated_code():
    code = (
        "def f0(x: int, s: str):\n"
        "    if x > 0:\n"
        "        return x + len(s)\n"
        "    return x\n"
        "def f1(y: int):\n"
        "    return y\n"
        "def endpoint(x: int, s: str):\n"
        "    return f0(x, s)\n"
    )
    mutations = list(organic_mutations(code))

    assert mutations
    assert all(candidate != code for _, candidate in mutations)
    assert all("def f0" in candidate and "def endpoint" in candidate for _, candidate in mutations)
