import random

from reasoning_core.template import Task

from reasoning_core.tasks.generated.wave4.s41_burrows_wheeler.burrows_wheeler import (
    BurrowsWheeler,
    _bwt_forward,
    _bwt_inverse,
    _is_valid_forward,
    TERMINATOR,
)


def test_forward_inverse_roundtrip():
    for s in ["abac", "banana", "mississippi", "cab"]:
        assert _bwt_inverse(_bwt_forward(s + "$")) == s + "$"


def test_config_difficulty():
    cfg = BurrowsWheeler().config
    base_max = cfg.max_len
    cfg.set_level(5)
    assert cfg.max_len >= base_max


def test_generate_and_score():
    task = BurrowsWheeler()
    for _ in range(200):
        ex = task.generate_entry()
        assert _bwt_inverse_assert(ex)
        assert task.score_answer(ex.answer, ex) == 1.0
        assert task.score_answer("zzzz", ex) == 0.0
        assert task.score_answer("", ex) == 0.0


def test_all_levels_generate():
    task = BurrowsWheeler()
    for level in (0, 1, 3, 5, 6):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_entry()
            assert _bwt_inverse_assert(ex)
            assert task.score_answer(ex.answer, ex) == 1.0


def test_score_rejects_junk():
    task = BurrowsWheeler()
    ex = task.generate_entry()
    assert task.score_answer("$" + ex.answer, ex) == 0.0
    assert task.score_answer("random-string", ex) == 0.0
    assert task.score_answer(12345, ex) == 0.0


def _bwt_inverse_assert(ex):
    meta = ex.metadata
    s = meta.payload["string"]
    if meta.mode == "forward":
        return _is_valid_forward(s + TERMINATOR) and _bwt_forward(s + TERMINATOR) == ex.answer
    else:
        return _bwt_inverse(s).replace(TERMINATOR, "") == ex.answer
