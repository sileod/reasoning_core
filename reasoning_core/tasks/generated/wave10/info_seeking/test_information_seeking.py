import random

from reasoning_core.tasks.generated.wave10.information_seeking.information_seeking import (
    InfoSeeking,
    InfoSeekingConfig,
    _best_position,
    _group_counts,
)


def test_generate_and_score():
    random.seed(147658999)
    task = InfoSeeking()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_not_scored():
    random.seed(7)
    task = InfoSeeking()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("garbage", ex) < 1.0


def test_answer_reproduces_split():
    random.seed(11)
    task = InfoSeeking()
    for _ in range(20):
        ex = task.generate_example()
        p, d = ex.answer.split(":")
        p, d = int(p), int(d)
        assert d == ex.metadata.d
        assert p == ex.metadata.p
        assert _group_counts(ex.metadata.hypos, p) == d
        for q in range(ex.metadata.m):
            assert _group_counts(ex.metadata.hypos, q) <= d
            if q < p:
                assert _group_counts(ex.metadata.hypos, q) < d


def test_difficulty_changes():
    cfg = InfoSeekingConfig()
    cfg.set_level(0)
    base = (cfg.n_hypos, cfg.m_len)
    cfg.set_level(3)
    hi = (cfg.n_hypos, cfg.m_len)
    cfg.set_level(6)
    hi6 = (cfg.n_hypos, cfg.m_len)
    assert base[0] < hi[0] < hi6[0]
    assert base[1] < hi[1] < hi6[1]


def test_all_levels_generate():
    task = InfoSeeking()
    for lvl in range(0, 7):
        random.seed(100 + lvl)
        cfg = InfoSeekingConfig()
        cfg.set_level(lvl)
        task.config = cfg
        ex = task.generate_example()
        assert ex.answer


def test_best_position_helper():
    hypos = ["aa", "ab", "ba"]
    p, d = _best_position(hypos, 2)
    # pos0 -> {a,b} = 2, pos1 -> {a,b} = 2; tie -> smallest index 0
    assert (p, d) == (0, 2)
