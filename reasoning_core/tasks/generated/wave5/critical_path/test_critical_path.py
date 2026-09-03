import random

from reasoning_core.tasks.generated.wave5.s56_critical_path.critical_path_task import (
    CriticalPath, CriticalPathConfig, analyze,
)


def test_analyze_reference():
    n = 3
    durations = [3, 4, 5]
    deps = [[], [], [0, 1]]
    project, critical = analyze(n, deps, durations)
    assert project == 4 + 5
    assert len(critical) > 0


def test_gold_answer_duration():
    random.seed(1)
    t = CriticalPath()
    for _ in range(20):
        e = t.generate_entry()
        if e.metadata.question == 'duration':
            assert t.score_answer(e.answer, e) == 1.0


def test_gold_answer_critical():
    random.seed(2)
    t = CriticalPath()
    got_dur = False
    got_crit = False
    for _ in range(100):
        e = t.generate_entry()
        if e.metadata.question == 'duration':
            got_dur = True
            assert t.score_answer(e.answer, e) == 1.0
        else:
            got_crit = True
            assert t.score_answer(e.answer, e) == 1.0
    assert got_dur and got_crit


def test_critical_has_multiple():
    random.seed(3)
    t = CriticalPath()
    for _ in range(50):
        e = t.generate_entry()
        assert len(e.metadata.critical) >= 2


def test_junk_scores_zero():
    random.seed(4)
    t = CriticalPath()
    for _ in range(30):
        e = t.generate_entry()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("garbage", e) == 0.0


def test_all_levels():
    t = CriticalPath()
    for level in range(7):
        cfg = CriticalPathConfig()
        cfg.set_level(level)
        t.config = cfg
        e = t.generate_entry()
        assert t.score_answer(e.answer, e) == 1.0
