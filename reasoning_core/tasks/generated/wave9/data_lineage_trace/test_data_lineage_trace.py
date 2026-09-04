import random

from reasoning_core.tasks.generated.wave9.data_lineage_trace.data_lineage_trace import (
    DataLineageTrace,
    LineageConfig,
)


def _seed(level, s=1):
    random.seed(abs(hash((level, s))) % (2 ** 32))


def test_gold_scores_one():
    random.seed(895421081)
    task = DataLineageTrace()
    for level in (0, 2, 5):
        for _ in range(20):
            ex = task.generate_example(level=level)
            assert task.score_answer(ex.answer, ex) == 1.0


def test_answer_changes_across_examples():
    random.seed(895421081)
    task = DataLineageTrace()
    seen = set()
    for _ in range(30):
        ex = task.generate_example(level=0)
        seen.add(ex.answer)
    assert len(seen) >= 3


def test_junk_scores_zero():
    random.seed(895421081)
    task = DataLineageTrace()
    ex = task.generate_example(level=0)
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("not a number", ex) == 0.0


def test_answer_always_nonempty_valid_ids():
    random.seed(3)
    task = DataLineageTrace()
    for level in (0, 3, 6):
        for _ in range(30):
            ex = task.generate_example(level=level)
            assert ex.answer != "NONE"
            ids = [int(x) for x in ex.answer.split(", ")]
            assert ids == sorted(ids)
            assert len(ids) == len(set(ids))
            for i in ids:
                assert i >= 1000


def test_level_changes_config():
    c = LineageConfig()
    c.set_level(0)
    n0 = c.n_rows
    c.set_level(5)
    assert c.n_rows > n0


def test_multi_line_extra_spaces():
    random.seed(895421081)
    task = DataLineageTrace()
    ex = task.generate_example(level=0)
    if ex.answer != "NONE":
        assert task.score_answer(ex.answer.replace(", ", ",  "), ex) == 1.0
