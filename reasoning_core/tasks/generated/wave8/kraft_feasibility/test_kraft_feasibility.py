from fractions import Fraction

from reasoning_core.tasks.generated.wave8.kraft_feasibility.kraft_feasibility import (
    KraftFeasibility,
)


def test_gold_scores_one_all_levels():
    task = KraftFeasibility()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(8):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0


def test_answer_is_reduced_fraction_witness():
    task = KraftFeasibility()
    task.config.set_level(0)
    for _ in range(20):
        ex = task.generate_example()
        total = Fraction(ex.answer)
        lengths = ex.metadata.lengths
        expected = sum(Fraction(1, 2 ** ell) for ell in lengths)
        assert total == expected
        assert 0 < total <= len(lengths)


def test_junk_and_empty_not_full():
    task = KraftFeasibility()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("not a number", ex) < 1.0
    assert task.score_answer("abc/def", ex) < 1.0


def test_wrong_answer_scores_zero():
    task = KraftFeasibility()
    for _ in range(20):
        ex = task.generate_example()
        total = Fraction(ex.answer)
        wrong = total + Fraction(1, 2)
        assert task.score_answer(str(wrong), ex) == 0.0


def test_decimal_form_accepted():
    task = KraftFeasibility()
    for _ in range(20):
        ex = task.generate_example()
        total = Fraction(ex.answer)
        dec = float(total)
        assert task.score_answer(str(dec), ex) == 1.0


def test_difficulty_changes_size():
    task = KraftFeasibility()
    task.config.set_level(0)
    n0 = int(task.config.n_len)
    task.config.set_level(5)
    n5 = int(task.config.n_len)
    assert n5 > n0
