import random
import ast

from reasoning_core.tasks.generated.wave8.left_coset_identification.left_coset_identification import (
    LeftCosetIdentification,
    LeftCosetConfig,
)


def _canonical(coset, n):
    return [c % n for c in sorted(set(coset))]


def test_gold_scores_one():
    task = LeftCosetIdentification()
    for level in range(7):
        task.config.set_level(level)
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_coset_matches_definition():
    task = LeftCosetIdentification()
    for level in range(7):
        task.config.set_level(level)
        e = task.generate_example()
        n = e.metadata["n"]
        a = e.metadata["a"]
        g = e.metadata["g"]
        subgroup = sorted({(k * a) % n for k in range(n)})
        expected = sorted({(s + g) % n for s in subgroup})
        ans = ast.literal_eval(e.answer)
        assert ans == expected


def test_wrong_and_random_score_zero():
    task = LeftCosetIdentification()
    task.config.set_level(2)
    e = task.generate_example()
    n = e.metadata["n"]
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0
    wrong = ast.literal_eval(e.answer)[:]
    assert task.score_answer(str(wrong + [n]), e) == 0.0


def test_difficulty_increases():
    c0 = LeftCosetConfig()
    c6 = LeftCosetConfig()
    c0.set_level(0)
    c6.set_level(6)
    assert c6.n_group > c0.n_group
