import random

from reasoning_core.tasks.generated.wave8.unix_mode_permission.unix_mode_permission import (
    UnixModePermission,
    user_granted,
    parse_granted,
)


def test_gold_scores_one_all_levels():
    task = UnixModePermission()
    for L in (0, 2, 5):
        task.config.set_level(L)
        for _ in range(15):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0


def test_parse_granted():
    assert parse_granted("none") == ()
    assert parse_granted("None") == ()
    assert parse_granted("1 3 7") == (1, 3, 7)


def test_parse_granted_rejects_empty():
    import pytest
    for bad in ("", None, "  ", "a b", "1 x"):
        with pytest.raises(ValueError):
            parse_granted(bad)


def test_garbage_and_partial():
    task = UnixModePermission()
    task.config.set_level(2)
    ex = task.generate_example()
    assert task.score_answer("garbage", ex) == 0.0
    assert task.score_answer(None, ex) == 0.0
    assert task.score_answer("", ex) == 0.0
    granted = parse_granted(ex.answer)
    if len(granted) >= 2:
        assert task.score_answer(" ".join(str(g) for g in granted[:1]), ex) == 0.5


def test_answer_not_surface_readable():
    task = UnixModePermission()
    for _ in range(30):
        ex = task.generate_example()
        text = ex.metadata.payload["Users"]
        assert ex.answer not in text.split()


def test_difficulty_increases():
    cfg = UnixModePermission.config_cls()
    base = cfg.n_users
    cfg.set_level(1)
    assert cfg.n_users >= base


def test_answer_spread():
    task = UnixModePermission()
    answers = []
    for L in (0, 2, 4):
        task.config.set_level(L)
        for _ in range(25):
            answers.append(task.generate_example().answer)
    top = max(set(answers), key=answers.count)
    assert answers.count(top) / len(answers) < 0.4
    assert len(set(answers)) > 5


def test_user_granted_logic():
    mode = 0o654  # owner=rw, group=r-x, other=r
    assert user_granted(mode, 100, 1, [200], "read") is True
    assert user_granted(mode, 200, 1, [200], "read") is True
    assert user_granted(mode, 200, 1, [200], "write") is False
    assert user_granted(mode, 300, 1, [999], "write") is False
    assert user_granted(mode, 300, 1, [999], "execute") is False
    mode2 = 0o777
    assert user_granted(mode2, 100, 1, [999], "execute") is True
