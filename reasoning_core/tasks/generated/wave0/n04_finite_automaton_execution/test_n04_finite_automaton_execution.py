import random

from reasoning_core.tasks.generated.wave0.n04_finite_automaton_execution.n04_finite_automaton_execution import (
    FiniteAutomatonExecution,
)


def test_roundtrip_all_levels():
    t = FiniteAutomatonExecution()
    for L in (0, 2, 5):
        t.config.set_level(L)
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers():
    t = FiniteAutomatonExecution()
    t.config.set_level(2)
    e = t.generate_example()
    gt = int(e.answer)
    assert t.score_answer(str(gt + 1), e) == 0.0
    assert t.score_answer("garbage", e) == 0.0


def test_answer_varies():
    t = FiniteAutomatonExecution()
    answers = set()
    for _ in range(40):
        t.config.set_level(3)
        answers.add(t.generate_example().answer)
    assert len(answers) > 5


def test_no_crash_non_numeric():
    t = FiniteAutomatonExecution()
    t.config.set_level(0)
    e = t.generate_example()
    assert t.score_answer(None, e) == 0.0


def test_answer_distribution_spread():
    t = FiniteAutomatonExecution()
    answers = []
    for L in (0, 2, 4):
        t.config.set_level(L)
        for _ in range(25):
            answers.append(int(t.generate_example().answer))
    top = max(set(answers), key=answers.count)
    assert answers.count(top) / len(answers) < 0.4


def test_word_matches_alphabet():
    t = FiniteAutomatonExecution()
    for L in (0, 2, 5):
        t.config.set_level(L)
        e = t.generate_example()
        word = e.metadata.payload["word"].split("'")[1]
        for c in word:
            assert c in e.metadata.payload["alphabet"][1:-1].replace(",", "").strip(), L
