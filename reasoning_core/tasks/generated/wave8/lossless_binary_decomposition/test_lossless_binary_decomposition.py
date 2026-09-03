import pytest

from reasoning_core.tasks.generated.wave8.lossless_binary_decomposition.lossless_binary_decomposition import (
    LosslessBinaryDecomposition,
    LosslessBinaryConfig,
    _fd_closure,
    _is_lossless,
)


@pytest.fixture(scope="module")
def task():
    return LosslessBinaryDecomposition()


def test_generate_and_score_lossless(task):
    for _ in range(40):
        ex = task.generate_example()
        assert ex.answer.startswith("lossless:") or ex.answer.startswith("lossy:")
        assert task.score_answer(ex.answer, ex) == 1.0


def test_wrong_direction_scores_zero():
    task = LosslessBinaryDecomposition()
    ex = task.generate_example()
    lossless = ex.metadata["lossless"]
    fake = "lossless:{}->P1 via {A}->{B}" if not lossless else "lossy:{} misses A"
    assert task.score_answer(fake, ex) < 1.0


def test_empty_and_junk_score_zero():
    task = LosslessBinaryDecomposition()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("asdf qwer", ex) < 1.0


def test_closure_basic():
    fds = [(frozenset([0]), frozenset([1])), (frozenset([1]), frozenset([2]))]
    assert _fd_closure(frozenset([0]), fds) == frozenset([0, 1, 2])


def test_is_lossless_criterion():
    # X={0} -> Y={1} makes {01,02} lossless
    share = frozenset([0])
    p1 = frozenset([0, 1])
    p2 = frozenset([0, 2])
    fds = [(share, frozenset([1]))]
    assert _is_lossless(p1, p2, fds)
    assert not _is_lossless(p1, p2, [])
    # but if the fd goes to 2 instead of 1, still lossless (X -> Z)
    fds2 = [(share, frozenset([2]))]
    assert _is_lossless(p1, p2, fds2)


def test_difficulty_changes_config():
    cfg = LosslessBinaryConfig()
    cfg0 = LosslessBinaryConfig()
    cfg.set_level(0)
    cfg5 = LosslessBinaryConfig()
    cfg5.set_level(5)
    assert cfg5.n_attrs >= cfg0.n_attrs


def test_metadata_json_roundtrip(task):
    import json

    ex = task.generate_example()
    d = dict(ex.metadata)
    s = json.dumps(d)
    back = json.loads(s)
    assert set(["attributes", "fds", "p1", "p2", "answer", "lossless"]) <= set(back)
