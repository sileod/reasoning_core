import pytest

from reasoning_core.tasks.generated.wave0.n07_max_flow_min_cut.max_flow_min_cut import MaxFlowMinCut


@pytest.mark.parametrize("level", [0, 2, 5])
def test_round_trip(level):
    t = MaxFlowMinCut()
    t.config.set_level(level)
    entry = t.generate_example()
    assert t.score_answer(entry.answer, entry) == 1.0
    assert t.render_prompt(entry.metadata)


@pytest.mark.parametrize("level", [0, 2, 5])
def test_validates(level):
    t = MaxFlowMinCut()
    t.config.set_level(level)
    t.validate()


def test_garbage_scores_zero():
    t = MaxFlowMinCut()
    entry = t.generate_example()
    assert t.score_answer("not a number", entry) == 0.0
    assert t.score_answer(None, entry) == 0.0
    assert t.score_answer("5", entry) == 0.0
    assert t.score_answer("5 0,9,99", entry) == 0.0
