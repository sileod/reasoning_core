import random

from reasoning_core.tasks.generated.wave9.three_valued_logic_propagation. \
    three_valued_logic_propagation import (
        ThreeValuedLogicPropagation, _not, _and, _or, _eval_expr, TR, FA, UN,
    )


def test_not():
    assert _not(TR) == FA
    assert _not(FA) == TR
    assert _not(UN) == UN


def test_and():
    assert _and(FA, TR) == FA
    assert _and(TR, TR) == TR
    assert _and(TR, UN) == UN
    assert _and(UN, UN) == UN
    assert _and(UN, FA) == FA


def test_or():
    assert _or(TR, UN) == TR
    assert _or(FA, FA) == FA
    assert _or(FA, UN) == UN
    assert _or(FA, TR) == TR


def test_eval_expr():
    assert _eval_expr(("and", ("not", "True"), "True")) == FA
    assert _eval_expr(("or", "Unknown", "False")) == UN
    assert _eval_expr(("and", "True", "Unknown")) == UN


def test_modes_occur():
    random.seed(1851018107)
    task = ThreeValuedLogicPropagation()
    seen = set()
    for _ in range(300):
        entry = task.generate_entry()
        seen.add(entry.metadata.mode)
        assert isinstance(entry.answer, str)
        assert entry.answer
    assert seen == {"net", "hybrid"}


def test_score_gold_and_junk():
    task = ThreeValuedLogicPropagation()
    random.seed(1851018107)
    for _ in range(50):
        entry = task.generate_entry()
        assert task.score_answer(entry.answer, entry) == 1.0
        assert task.score_answer("", entry) == 0.0
        assert task.score_answer("nonsense", entry) == 0.0
        assert task.score_answer("True; True; True", entry) in (0.0, 1.0)


def test_difficulty_changes():
    task = ThreeValuedLogicPropagation()
    base = int(task.config.n_gates)
    task.config.set_level(3)
    assert int(task.config.n_gates) > base
