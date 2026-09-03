import random
import pytest

from reasoning_core.tasks.generated.wave8.virtual_address_translation.virtual_address_translation import (
    VirtualAddressTranslation,
)


@pytest.fixture
def task():
    return VirtualAddressTranslation()


def test_gold_scores_one(task):
    random.seed(28548294)
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_zero(task):
    random.seed(5)
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("garbage", ex) == 0.0
    assert task.score_answer(None, ex) == 0.0


def test_answer_domain(task):
    random.seed(11)
    for _ in range(30):
        ex = task.generate_example()
        if ex.answer == "page fault":
            assert ex.metadata.answer_is_fault is True
        else:
            val = int(ex.answer)
            assert val >= 0
            assert ex.metadata.answer_is_fault is False


def test_difficulty_changes(task):
    base = task.config.set_level(0)
    c0 = VirtualAddressTranslation().config
    c6 = VirtualAddressTranslation().config
    c6.set_level(6)
    assert c6 != c0


def test_labels_spread(task):
    random.seed(99)
    seen = set()
    nf = f = 0
    for _ in range(60):
        ex = task.generate_example()
        seen.add(ex.answer)
        if ex.answer == "page fault":
            f += 1
        else:
            nf += 1
    assert len(seen) > 20
    assert f >= 1
    assert nf >= 1
