import random
from reasoning_core.tasks.generated.wave9.memory_allocator_trace.memory_allocator_trace import (
    MemoryAllocatorTrace,
    _simulate,
    _layout_str,
    MemoryAllocatorConfig,
)


def test_example_roundtrip():
    random.seed(1)
    task = MemoryAllocatorTrace()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_answer_varies():
    random.seed(2)
    task = MemoryAllocatorTrace()
    answers = set()
    for _ in range(30):
        x = task.generate_example()
        answers.add(x.answer)
    assert len(answers) > 5


def test_simulate_matches_gold():
    random.seed(3)
    task = MemoryAllocatorTrace()
    for _ in range(20):
        x = task.generate_example()
        spans = _simulate(x.metadata.blocks, x.metadata.ops, x.metadata.strategy)
        assert _layout_str(spans) == x.answer


def test_free_reuses_coalesced():
    blocks = [[0, 5]]
    cmd = [("alloc", 2), ("alloc", 2), ("free", 0)]
    spans = _simulate(blocks, cmd, "first_fit")
    # after alloc2, alloc2, free0 -> free [0,2) and [4,5) remain -> two spans
    assert spans == [[0, 2], [4, 5]]


def test_junk_scores_zero():
    task = MemoryAllocatorTrace()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("garbage", x) == 0.0


def test_importance_rejection_no_fail_at_all_levels():
    for level in range(7):
        cfg = MemoryAllocatorConfig()
        cfg.set_level(level)
        cfg.seed = level
        task = MemoryAllocatorTrace()
        task.config = cfg
        x = task.generate_entry()
        assert x.answer != ""
