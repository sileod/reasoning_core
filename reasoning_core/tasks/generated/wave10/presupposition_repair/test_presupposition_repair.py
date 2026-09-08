import pytest

from reasoning_core.tasks.generated.wave10.presupposition_repair.presupposition_repair import (
    PresuppositionRepair,
    PresuppositionRepairV2Config,
)


def _fact_value(fact, target):
    for tok in fact.split():
        if tok.isdigit():
            if target in fact.split():
                return int(tok)
    return None


def _check_gold(entry):
    gold = int(entry.answer)
    meta = entry.metadata
    a = meta.a
    b = meta.b
    va = vb = None
    for f in meta.facts:
        tokens = f.split()
        if tokens[0] == a or (a in tokens):
            v = _fact_value(f, a)
            if v is not None:
                va = v
        if tokens[0] == b or (b in tokens):
            v = _fact_value(f, b)
            if v is not None:
                vb = v
    assert va is not None and vb is not None
    if ">=" in meta.premise:
        true = va >= vb
    elif "==" in meta.premise:
        true = va == vb
    else:
        true = va < vb
    assert meta.premise_true == true
    expected = va + vb if meta.premise_true else va + vb - 3
    if expected < 0:
        expected = va + vb
    assert gold == expected


def test_gold_scores_one():
    task = PresuppositionRepair()
    for _ in range(20):
        task.config = PresuppositionRepairV2Config()
        entry = task.generate_example()
        _check_gold(entry)
        assert task.score_answer(entry.answer, entry) == 1.0


def test_junk_scores_zero():
    task = PresuppositionRepair()
    task.config = PresuppositionRepairV2Config()
    entry = task.generate_example()
    assert task.score_answer("", entry) < 1.0
    assert task.score_answer("garbage", entry) < 1.0


def test_levels_difficulty():
    cfg = PresuppositionRepairV2Config()
    base = cfg.facts
    cfg.set_level(5)
    assert cfg.facts >= base


def test_all_levels_generate():
    task = PresuppositionRepair()
    for level in (0, 1, 2, 3, 4, 5, 6):
        task.config = PresuppositionRepairV2Config()
        task.config.set_level(level)
        entry = task.generate_example()
        _check_gold(entry)
        assert task.score_answer(entry.answer, entry) == 1.0
