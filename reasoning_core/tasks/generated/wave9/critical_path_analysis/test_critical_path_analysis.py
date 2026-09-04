import random

from reasoning_core.tasks.generated.wave9.critical_path_analysis.critical_path_analysis import CriticalPathAnalysis


def test_smoke():
    random.seed(1)
    t = CriticalPathAnalysis()
    x = t.generate_example()
    assert t.score_answer(x.answer, x) == 1.0
    assert t.score_answer("", x) < 1.0
    assert t.score_answer("garbage zz", x) < 1.0


def test_difficulty_changes():
    t = CriticalPathAnalysis()
    base = t.config.n_tasks
    for level in (2, 5):
        t.config.set_level(level)
        assert t.config.n_tasks > base
