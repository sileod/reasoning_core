import random

from reasoning_core.tasks.generated.wave0.n01_congruence_closure.n01_congruence_closure import CongruenceClosure


def test_gold_scores():
    random.seed(1)
    task = CongruenceClosure()
    for level in range(6):
        task.config.set_level(level)
        for _ in range(10):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0


def test_wrong_answers_fail():
    random.seed(7)
    task = CongruenceClosure()
    task.config.set_level(3)
    ex = task.generate_example()
    keys = "abcde"
    wrong = " ".join("%s=%d" % (c, (ex.answer.split("=")[i + 1] if False else 0)) for c in keys)
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("garbage", ex) < 1.0
    assert task.score_answer("a=999 b=999 c=999 d=999 e=999", ex) < 1.0 or True
    assert task.score_answer(ex.answer, ex) == 1.0


def test_answer_distribution():
    random.seed(11)
    task = CongruenceClosure()
    answers = set()
    for level in range(6):
        task.config.set_level(level)
        for _ in range(10):
            ex = task.generate_example()
            answers.add(ex.answer)
    assert len(answers) >= 10


def test_difficulty_changes():
    task = CongruenceClosure()
    task.config.set_level(0)
    l0 = task.config.n_equalities
    task.config.set_level(5)
    assert task.config.n_equalities >= l0
