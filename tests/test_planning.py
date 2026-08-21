from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from unified_planning.shortcuts import BoolType, Object, Problem, UserType

import reasoning_core.tasks.planning as planning
from reasoning_core.tasks.planning import translate


def test_random_solve_phases_out_by_level_three():
    config = planning.PlanningConfig()
    assert [config.set_level(level).pure_random_proba for level in range(5)] == pytest.approx(
        [0.12, 0.08, 0.04, 0, 0]
    )
    assert [config.set_level(level).optimal_relabel for level in range(5)] == [True, True, True, False, False]


def test_high_level_planning_adds_sparse_plan_cues(monkeypatch):
    actions = [SimpleNamespace(action=SimpleNamespace(name=f"action_{i}")) for i in range(7)]
    plan = SimpleNamespace(actions=actions)
    writer = SimpleNamespace(get_problem=lambda: "problem", get_domain=lambda: "domain")
    task = planning.Planning(planning.PlanningConfig().set_level(3))

    monkeypatch.setattr(planning, "generate_domain", lambda *args, **kwargs: object())
    monkeypatch.setattr(planning, "generate_planted_problem", lambda *args, **kwargs: (object(), plan, "medium"))
    monkeypatch.setattr(planning, "format_plan", lambda _: "plan")
    monkeypatch.setattr(planning, "translate", lambda _: "problem")
    monkeypatch.setattr(planning, "PDDLWriter", lambda _: writer)
    monkeypatch.setattr(planning, "make_cot", lambda *args: "trace")
    monkeypatch.setattr(task, "score_answer", lambda *args: 1)

    entry = task.generate_entry()
    assert entry.metadata.plan_cue == {
        "length": 7,
        "steps": [{"step": 3, "action": "action_2"}, {"step": 6, "action": "action_5"}],
    }
    prompt = task.render_prompt(entry.metadata)
    assert "Cue: exactly 7 actions; step 3 uses action_2; step 6 uses action_5." in prompt
    assert "Hint:" not in prompt


def test_score_rejects_plan_that_misses_cue(monkeypatch):
    reader = Mock()
    reader.parse_problem_string.return_value = SimpleNamespace(kind=object())
    reader.parse_plan_string.return_value = SimpleNamespace(
        actions=[SimpleNamespace(action=SimpleNamespace(name="wrong"))]
    )
    monkeypatch.setattr(planning, "PDDLReader", lambda: reader)
    meta = {"domain_pddl": "d", "problem_pddl": "p",
            "plan_cue": {"length": 1, "steps": [{"step": 1, "action": "right"}]}}

    reward = planning.Planning().score_answer("wrong()", {"metadata": meta})
    assert reward == 0
    assert reward.tag == "plan cue mismatch"


def test_translate_uses_closed_world_initial_state():
    item = UserType("item")
    problem = Problem("closed-world-state")
    active = problem.add_fluent(
        "active",
        BoolType(),
        item=item,
        default_initial_value=True,
    )
    first = Object("first", item)
    second = Object("second", item)
    problem.add_objects([first, second])
    problem.set_initial_value(active(first), True)
    problem.set_initial_value(active(second), False)

    prompt = translate(problem)

    assert "Default value:" not in prompt
    assert "True values: active(first)" in prompt
    assert "active(second)" not in prompt
    assert "All facts not listed under True values are false." in prompt


def test_generate_entry_does_not_reseed_global_random(monkeypatch):
    class StopGeneration(BaseException):
        pass

    def stop_generation():
        raise StopGeneration

    seed = Mock()
    monkeypatch.setattr(planning.random, "seed", seed)
    monkeypatch.setattr(planning.random, "random", stop_generation)
    monkeypatch.setattr(planning, "generate_domain", lambda *args, **kwargs: object())

    with pytest.raises(StopGeneration):
        planning.Planning().generate_entry()

    seed.assert_not_called()


def test_generate_entry_retries_finalization_errors(monkeypatch):
    plan = SimpleNamespace(actions=[object()] * 3)
    solution = SimpleNamespace(plan=plan)
    writer = SimpleNamespace(get_problem=lambda: "problem", get_domain=lambda: "domain")
    task = planning.Planning(planning.PlanningConfig(pure_random_proba=1.0))

    monkeypatch.setattr(planning, "generate_domain", lambda *args, **kwargs: object())
    monkeypatch.setattr(planning, "generate_problem", lambda *args, **kwargs: object())
    monkeypatch.setattr(planning, "solve", lambda *args, **kwargs: solution)
    monkeypatch.setattr(planning, "format_plan", lambda _: "action_0(object_1)")
    translate = Mock(side_effect=[Exception("action_1_parameter0_type_0"), "problem"])
    monkeypatch.setattr(planning, "translate", translate)
    monkeypatch.setattr(planning, "PDDLWriter", lambda _: writer)
    monkeypatch.setattr(planning, "make_cot", lambda *args: "trace")
    monkeypatch.setattr(task, "score_answer", lambda *args: 1)

    entry = task.generate_entry()

    assert entry.answer == "action_0(object_1)"
    assert translate.call_count == 2
