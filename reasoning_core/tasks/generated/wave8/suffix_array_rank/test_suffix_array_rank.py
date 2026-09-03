import random

random.seed(3317076631)

from reasoning_core.tasks.generated.wave8.suffix_array_rank.suffix_array_rank import (
    SuffixRank,
    build_suffix_array,
)


def test_build_suffix_array():
    assert build_suffix_array("banana") == sorted(range(6), key=lambda i: "banana"[i:])
    assert build_suffix_array("a") == [0]
    assert build_suffix_array("aaa") == [2, 1, 0]


def test_roundtrip_all_levels():
    task = SuffixRank()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(30):
            entry = task.generate_example()
            assert task.score_answer(entry.answer, entry) == 1.0
            s = entry.metadata.s
            idx = entry.metadata.index
            assert int(entry.answer) == sorted(s[i:] for i in range(len(s))).index(s[idx:])


def test_balance_and_range():
    task = SuffixRank()
    answers = set()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(40):
            entry = task.generate_example()
            answers.add(int(entry.answer))
    assert len(answers) >= 5


def test_junk_scores_zero():
    task = SuffixRank()
    task.config.set_level(2)
    entry = task.generate_example()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("banana", entry) == 0.0
    assert task.score_answer("3.5", entry) == 0.0
