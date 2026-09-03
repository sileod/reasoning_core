import random

from reasoning_core.tasks.generated.wave8.mvcc_visibility.mvcc_visibility import (
    MVCCVisibility,
    score_answer,
)


def test_gold_scores_one_row():
    random.seed(1305167045)
    task = MVCCVisibility()
    for _ in range(50):
        e = task.generate_example()
        assert score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    task = MVCCVisibility()
    for _ in range(20):
        e = task.generate_example()
        assert score_answer("", e) < 1.0
        assert score_answer("garbage", e) < 1.0
        assert score_answer("banana", e) < 1.0


def test_wrong_answer_scores_zero():
    random.seed(7)
    task = MVCCVisibility()
    e = task.generate_example()
    ans = e.answer
    assert score_answer(ans + " ", e) == 1.0
    assert score_answer("(999,999)", e) < 1.0


def test_control_flow_visibility():
    # direct verifier check: newest begin <= T wins
    from reasoning_core.tasks.generated.wave8.mvcc_visibility.mvcc_visibility import (
        _visible,
    )
    versions = [(1, 5), (3, 9), (10, 15)]
    assert _visible(versions, 4) == (3, 9)
    assert _visible(versions, 1) == (1, 5)
    assert _visible(versions, 2) == (1, 5)
    assert _visible(versions, 8) == (3, 9)
    assert _visible(versions, 9) == (3, 9)
    assert _visible(versions, 10) == (10, 15)
    assert _visible(versions, 0) is None


def test_all_levels_generate():
    task = MVCCVisibility()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(10):
            e = task.generate_example()
            assert score_answer(e.answer, e) == 1.0
