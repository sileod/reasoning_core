import copy
import json
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


def test_high_level_planning_has_one_cue_conditioned_solution():
    planning.random.seed(3)
    task = planning.Planning()
    entry = task.generate_example(level=3, max_tokens=0)

    assert entry.metadata.engine == "bounded-strips-v1"
    assert entry.metadata.solution_count == 1
    assert entry.metadata.plan_cue.length == entry.metadata.horizon == 6
    assert 1 <= len(entry.metadata.plan_cue.steps) <= 2
    assert entry.metadata._task_version == 2
    assert "verif_cot" not in entry.metadata
    assert entry.answer not in json.dumps(entry.metadata)
    assert "Cue: exactly 6 actions" in entry.prompt
    assert task.score_answer(entry.answer, entry) == 1

    cue = entry.metadata.plan_cue.steps[0]
    chosen = next(a for a in entry.metadata.actions if a.call == cue.action)
    alternative = next(a for a in entry.metadata.actions if a.call != chosen.call and
                       (a.pre_true, a.pre_false, a.add, a.delete) ==
                       (chosen.pre_true, chosen.pre_false, chosen.add, chosen.delete))
    answer = entry.answer.splitlines(); answer[cue.step - 1] = alternative.call
    assert task.score_answer("\n".join(answer), entry) == 0
    uncued = copy.deepcopy(entry.metadata); uncued.plan_cue.steps = []
    assert task.score_answer("\n".join(answer), {"metadata": uncued}) == 1


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
    monkeypatch.setattr(task, "score_answer", lambda *args: 1)

    entry = task.generate_entry()

    assert entry.answer == "action_0(object_1)"
    assert "verif_cot" not in entry.metadata
    assert translate.call_count == 2
