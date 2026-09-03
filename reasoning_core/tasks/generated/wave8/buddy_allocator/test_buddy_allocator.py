import random

from reasoning_core.tasks.generated.wave8.buddy_allocator.buddy_allocator import (
    BuddyAllocator, _format_sizes, _parse_multiset, Buddy,
)


def _sizes_list(answer):
    if answer.strip() == "empty":
        return []
    return _parse_multiset(answer)


def test_gold_scores_one():
    random.seed(1)
    task = BuddyAllocator()
    for _ in range(300):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    random.seed(2)
    task = BuddyAllocator()
    for _ in range(50):
        e = task.generate_example()
        assert task.score_answer("", e) == 0.0
        assert task.score_answer("garbage", e) == 0.0
        assert task.score_answer("0x0", e) == 0.0
        assert task.score_answer("1x0", e) == 0.0


def test_answer_domain():
    random.seed(3)
    task = BuddyAllocator()
    for _ in range(500):
        e = task.generate_example()
        sizes = _sizes_list(e.answer)
        assert all(c >= 1 and s >= 1 and (s & (s - 1)) == 0 for c, s in sizes)
        nfree = sum(c * s for c, s in sizes)
        assert nfree <= (1 << e.metadata.order)


def test_free_alloc_consistency():
    random.seed(4)
    task = BuddyAllocator()
    for _ in range(500):
        e = task.generate_example()
        sizes = _sizes_list(e.answer)
        nfree = sum(c * s for c, s in sizes)
        alloc_total = (1 << e.metadata.order) - nfree
        assert alloc_total >= 0
        assert alloc_total % 1 == 0


def test_distinct_answers():
    random.seed(5)
    task = BuddyAllocator()
    seen = set()
    for _ in range(2000):
        e = task.generate_example()
        seen.add(e.answer)
    assert len(seen) > 40, f"only {len(seen)} distinct answers"


def test_buddy_merge_is_buddy_aware():
    b = Buddy(3)
    b.free = {2: 2, 6: 2}
    b.alloc = {0: 2, 4: 2}
    b.free_block(0)
    assert sorted(b.free_sizes()) == [2, 4], "block at 6 must not merge with non-buddy block at 0 (2 via 0's buddy 2)"

    b2 = Buddy(3)
    b2.free = {2: 2, 6: 2}
    b2.alloc = {0: 2, 4: 2}
    b2.free_block(0)
    b2.free_block(4)
    assert sorted(b2.free_sizes()) == [8], "freeing buddies must merge to single full block"


def test_parse_roundtrip():
    assert _format_sizes([4, 4, 2]) == "2x4; 1x2"
    out = _format_sizes([2, 2, 1, 1])
    assert out == "2x2; 2x1"
    assert _format_sizes([]) == "empty"
