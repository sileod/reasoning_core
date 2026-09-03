import random

from reasoning_core.tasks.generated.wave8.kmp_prefix_value.kmp_prefix_value import (
    KmpPrefixValue,
    KmpPrefixValueConfig,
    _prefix_function,
)


def test_generate_and_score():
    random.seed(1)
    task = KmpPrefixValue()
    for _ in range(50):
        entry = task.generate_entry()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_answer_is_valid_prefix_value():
    random.seed(7)
    task = KmpPrefixValue()
    for _ in range(50):
        entry = task.generate_entry()
        pos = entry.metadata["position"]
        value = int(entry.answer)
        pattern = entry.metadata["pattern"]
        pi = _prefix_function([ord(c) - 97 for c in pattern])
        assert 0 <= value <= pos
        assert pi[pos] == value


def test_wrong_answers_fail():
    random.seed(3)
    task = KmpPrefixValue()
    for _ in range(30):
        entry = task.generate_entry()
        gold = int(entry.answer)
        wrong = gold + 1
        assert task.score_answer(str(wrong), entry) == 0.0


def test_empty_and_junk():
    random.seed(5)
    task = KmpPrefixValue()
    entry = task.generate_entry()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("abc", entry) == 0.0


def test_difficulty_changes():
    c = KmpPrefixValueConfig()
    c.set_level(0)
    n0 = c.n
    c.set_level(5)
    assert c.n > n0
