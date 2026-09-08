import copy
import json

import pytest

from reasoning_core.tasks.planning import Planning


@pytest.mark.parametrize("level", range(7))
def test_every_level_has_one_cue_conditioned_solution(level):
    task = Planning()
    entry = task.generate_example(level=level, max_tokens=0)

    assert entry.metadata.engine == "bounded-strips-v1"
    assert entry.metadata.solution_count == 1
    assert entry.metadata.plan_cue.length == entry.metadata.horizon == 3 + level
    assert entry.metadata._task_version == 3
    assert "verif_cot" not in entry.metadata
    assert entry.answer not in json.dumps(entry.metadata)
    assert f"Cue: exactly {3 + level} actions" in entry.prompt
    assert task.score_answer(entry.answer, entry) == 1


def test_cue_selects_between_equivalent_plans():
    task = Planning()
    for _ in range(20):
        entry = task.generate_example(level=3, max_tokens=0)
        if entry.metadata.plan_cue.steps:
            break
    cue = entry.metadata.plan_cue.steps[0]
    chosen = next(a for a in entry.metadata.actions if a.call == cue.action)
    alternative = next(a for a in entry.metadata.actions if a.call != chosen.call and
                       (a.pre_true, a.pre_false, a.add, a.delete) ==
                       (chosen.pre_true, chosen.pre_false, chosen.add, chosen.delete))
    answer = entry.answer.splitlines()
    answer[cue.step - 1] = alternative.call

    assert task.score_answer("\n".join(answer), entry) == 0
    uncued = copy.deepcopy(entry.metadata)
    uncued.plan_cue.steps = []
    assert task.score_answer("\n".join(answer), {"metadata": uncued}) == 1


def test_score_rejects_invalid_plan():
    task = Planning()
    entry = task.generate_example(level=0, max_tokens=0)

    assert task.score_answer("invented(action)", entry) == 0
