import random

from reasoning_core.tasks.generated.wave9.counterfactual_twin_model.counterfactual_twin_model import (
    CounterfactualTwinModel,
    _answer_formula,
    _predict,
)


def test_roundtrip_scores_gold():
    task = CounterfactualTwinModel()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(200):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0


def test_formula_matches_simulation():
    task = CounterfactualTwinModel()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(200):
            ex = task.generate_example()
            m = ex.metadata
            assert _answer_formula(m.n, m.coef, m.v, m.k, m.j, m.w) == _predict(
                m.n, m.coef, m.v, m.k, m.j, m.w
            )
            assert int(ex.answer) == _answer_formula(m.n, m.coef, m.v, m.k, m.j, m.w)


def test_rejection_of_wrong_answers():
    task = CounterfactualTwinModel()
    task.config.set_level(3)
    for _ in range(100):
        ex = task.generate_example()
        assert task.score_answer(str(int(ex.answer) + 1), ex) < 1.0
        assert task.score_answer(str(int(ex.answer) - 1), ex) < 1.0


def test_empty_and_junk():
    task = CounterfactualTwinModel()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("abc", ex) < 1.0
    assert task.score_answer(None, ex) < 1.0


def test_answer_is_integer():
    task = CounterfactualTwinModel()
    for level in (0, 3, 6):
        task.config.set_level(level)
        for _ in range(50):
            ex = task.generate_example()
            int(ex.answer)


def test_intervention_above_query_never_generated():
    task = CounterfactualTwinModel()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(100):
            ex = task.generate_example()
            assert ex.metadata.k < ex.metadata.j
