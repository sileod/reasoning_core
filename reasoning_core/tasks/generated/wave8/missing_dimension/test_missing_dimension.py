import random

from reasoning_core.tasks.generated.wave8.missing_dimension.missing_dimension import (
    MissingDimension,
    _parse_vector,
)


def test_parse_vector():
    assert _parse_vector("1,-1,2,0") == [1, -1, 2, 0]
    assert _parse_vector(" 0, 0, 0, 0 ") == [0, 0, 0, 0]
    assert _parse_vector("") is None
    assert _parse_vector("abc") is None
    assert _parse_vector("1,2,3") is None
    assert _parse_vector("1,2,3,4,5") is None


def test_gold_scores_one():
    task = MissingDimension()
    task.config.set_level(0)
    for _ in range(40):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0
        vec = _parse_vector(e.answer)
        assert len(vec) == 4
        assert vec != [0, 0, 0, 0]
        assert max(max(vec), -min(vec)) <= 8


def test_junk_scores_zero():
    task = MissingDimension()
    task.config.set_level(0)
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0
    assert task.score_answer("1,2,3,4", e) < 1.0 or _parse_vector(e.answer) == [1, 2, 3, 4]


def test_answer_spread():
    random.seed(0)
    task = MissingDimension()
    task.config.set_level(0)
    seen = set()
    for _ in range(200):
        e = task.generate_example()
        seen.add(e.answer)
    assert len(seen) >= 20, len(seen)


def test_answer_not_on_surface(monkeypatch=None):
    random.seed(1)
    task = MissingDimension()
    task.config.set_level(0)
    for _ in range(60):
        e = task.generate_example()
        ans_vec = _parse_vector(e.answer)
        payload = e.metadata.payload
        assert f",{e.answer}," not in str(payload)
        assert e.answer not in str(payload)


def test_difficulty_changes():
    task = MissingDimension()
    c0 = task.config.to_dict()
    task.config.set_level(3)
    c3 = task.config.to_dict()
    assert c0 != c3


def test_validate():
    MissingDimension().validate(n_samples=6)
