import random
import re

from reasoning_core.tasks.generated.wave8.c3_linearization.c3_linearization import (
    C3Linearization,
    C3LinearizationConfig,
    c3_linearize,
)


def test_mro_is_self_first_and_consistent():
    task = C3Linearization()
    for _ in range(50):
        ex = task.generate_example()
        q = ex.metadata.query.replace("class ", "")
        mro = ex.metadata.mro
        assert mro[0] == q
        assert len(mro) == len(set(mro))
        assert mro[-1] == "A"


def test_chain_linearization():
    mro = {"A": ["A"], "B": ["B", "A"], "C": ["C", "B", "A"]}
    assert c3_linearize("D", ["C"], mro) == ["D", "C", "B", "A"]


def test_diamond_linearization():
    mro = {"A": ["A"], "B": ["B", "A"], "C": ["C", "A"]}
    assert c3_linearize("D", ["B", "C"], mro) == ["D", "B", "C", "A"]


def test_score_accepts_gold_and_whitespace_variants():
    task = C3Linearization()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0
    norm = re.sub(r"\s+", "", ex.answer)
    compact = ",".join(ex.answer.split(", "))
    assert task.score_answer(compact, ex) == 1.0
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("Q, Z, Z", ex) == 0.0


def test_difficulty_changes_config():
    cfg = C3LinearizationConfig()
    cfg.set_level(1)
    assert cfg.level == 1


def test_reproducible_under_seed():
    random.seed(1493643473)
    a = C3Linearization().generate_example().answer
    random.seed(1493643473)
    b = C3Linearization().generate_example().answer
    assert a == b
