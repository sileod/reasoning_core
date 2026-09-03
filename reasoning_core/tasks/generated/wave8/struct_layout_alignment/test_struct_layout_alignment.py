import ast

from reasoning_core.tasks.generated.wave8.struct_layout_alignment.struct_layout_alignment import (
    StructLayoutAlignment,
    StructLayoutAlignmentConfig,
    compute_layout,
)


def test_roundtrip():
    task = StructLayoutAlignment()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0
        assert ast.literal_eval(ex.answer) is not None


def test_score_junk():
    task = StructLayoutAlignment()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("garbage", ex) < 1.0


def test_compute_layout():
    offsets, stride = compute_layout([(4, 4), (8, 8)], 'x86')
    assert offsets == [0, 8]
    assert stride == 16


def test_difficulty_changes():
    cfg = StructLayoutAlignmentConfig()
    cfg.set_level(0)
    n0 = int(cfg.n_fields)
    cfg.set_level(5)
    assert int(cfg.n_fields) > n0
