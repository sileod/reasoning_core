import random

from reasoning_core.tasks.generated.wave9.cache_replacement_trace.cache_replacement_trace import (
    CacheReplacementTrace,
    _simulate,
)


def test_generate_round_trip():
    random.seed(123)
    task = CacheReplacementTrace()
    for _ in range(50):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0
        assert task.score_answer("", e) < 1.0
        assert task.score_answer("junk", e) < 1.0


def test_simulator_counts_consistent():
    # total = hits + misses always equals length
    for _ in range(200):
        cap = random.randint(1, 5)
        alphabet = random.randint(cap, 10)
        length = random.randint(1, 30)
        accesses = [random.randint(0, alphabet - 1) for _ in range(length)]
        for policy in ('LRU', 'LFU', 'FIFO'):
            h, m, s = _simulate(policy, cap, accesses, 'fifo')
            assert h + m == length, (h, m, length)
            assert 0 <= h <= length
            assert len(s) <= cap


def test_fifo_oldest_first():
    # with distinct keys overflowing capacity, FIFO evicts earliest inserted
    cap = 2
    accesses = [0, 1, 2]
    h, m, s = _simulate('FIFO', cap, accesses, 'fifo')
    assert m == 3
    assert s == [1, 2]


def test_lru_eviction():
    cap = 2
    accesses = [0, 1, 0, 2]
    # after [0,1], then 0 hit (recency [1,0]), then 2 -> evict 1 (least recently used)
    h, m, s = _simulate('LRU', cap, accesses, 'lfu_key')
    assert s == [0, 2]


def test_lru_reference_behavior():
    # classic reference string
    cap = 3
    accesses = [1, 2, 3, 4, 1, 2]
    h, m, s = _simulate('LRU', cap, accesses, 'lfu_key')
    # 1,2,3 fill; 4 evicts 1; 1 evicts 2; 2 evicts 3 -> final {1,2,4}
    assert h == 0
    assert s == [1, 2, 4]


def test_lfu_eviction():
    # 0 used twice, 1 used once -> evict 1
    cap = 2
    accesses = [0, 1, 0, 2]
    h, m, s = _simulate('LFU', cap, accesses, 'lfu_key')
    assert s == [0, 2]


def test_lfu_tie_by_key():
    # 0 and 1 both accessed once, then 2 -> tie on freq, break by smallest key (0)
    cap = 2
    accesses = [0, 1, 2]
    h, m, s = _simulate('LFU', cap, accesses, 'lfu_key')
    assert s == [1, 2]


def test_lfu_tie_by_insertion():
    # 0 and 1 both accessed once, then 2 -> tie broken by earliest insertion (0)
    cap = 2
    accesses = [0, 1, 2]
    h, m, s = _simulate('LFU', cap, accesses, 'lfu_insertion')
    assert s == [1, 2]


def test_multi_level_generation():
    random.seed(7)
    task = CacheReplacementTrace()
    for level in range(7):
        task.config.set_level(level)
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_state_answer_format():
    random.seed(9)
    task = CacheReplacementTrace()
    for _ in range(30):
        e = task.generate_example()
        if e.metadata['query'] == 'state':
            assert e.answer.startswith('[') and e.answer.endswith(']')
