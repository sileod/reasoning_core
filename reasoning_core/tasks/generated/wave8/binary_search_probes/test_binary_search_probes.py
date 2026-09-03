import random

from reasoning_core.tasks.generated.wave8.binary_search_probes.binary_search_probes import (
    BinarySearchProbes,
    _binary_search_probes,
    _parse_answer,
)


def test_gold_scores_one():
    random.seed(1)
    task = BinarySearchProbes()
    for level in (0, 2, 5, 6):
        task.config.set_level(level)
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_wrong_does_not_score_one():
    random.seed(2)
    task = BinarySearchProbes()
    entry = task.generate_example()
    arr = entry.metadata.arr
    target = entry.metadata.target
    gold = _binary_search_probes(arr, target)
    if len(gold) > 1:
        wrong = str(gold[1:])
        assert task.score_answer(wrong, entry) == 0.0
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("not a list", entry) == 0.0


def test_probes_are_valid():
    random.seed(3)
    task = BinarySearchProbes()
    n_probes = 0
    for level in range(7):
        task.config.set_level(level)
        entry = task.generate_example()
        arr = entry.metadata.arr
        target = entry.metadata.target
        probes = _binary_search_probes(arr, target)
        assert all(0 <= p < len(arr) for p in probes)
        assert probes[0] == (len(arr) - 1) // 2
        n_probes += 1
    assert n_probes >= 7


def test_parse_answer_forms():
    assert _parse_answer("[4, 1, 2]") == [4, 1, 2]
    assert _parse_answer("4, 1, 2") == [4, 1, 2]
    assert _parse_answer("[]") == []
