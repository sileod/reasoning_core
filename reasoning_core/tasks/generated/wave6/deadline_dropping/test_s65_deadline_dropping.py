import random
from reasoning_core.tasks.generated.wave6.s65_deadline_dropping.s65_deadline_dropping import DeadlineDropping, DeadlineDroppingConfig

random.seed(65042966)


def test_generate_and_score():
    task = DeadlineDropping()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_zero():
    task = DeadlineDropping()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("nonsense", ex) == 0.0


def test_difficulty_changes():
    cfg = DeadlineDroppingConfig()
    cfg.set_level(0)
    base = cfg.n_jobs
    cfg.set_level(5)
    assert cfg.n_jobs >= base


def test_levels_generate():
    task = DeadlineDropping()
    for level in [0, 2, 3, 5, 6]:
        cfg = DeadlineDroppingConfig()
        cfg.set_level(level)
        task.config = cfg
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0
