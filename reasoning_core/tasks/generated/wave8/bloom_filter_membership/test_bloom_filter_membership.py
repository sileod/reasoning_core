import random
import importlib

import pytest

from reasoning_core.template import stochastic_rounding as sround

mod = importlib.import_module(
    "reasoning_core.tasks.generated.wave8.bloom_filter_membership.bloom_filter_membership"
)


def _new_task(level=0):
    t = mod.BloomFilterMembership()
    t.config.set_level(level)
    return t


def _compute_absent(metadata):
    m = metadata.payload["m"]
    a = metadata.payload["a"]
    b = metadata.payload["b"]
    k = metadata.payload["k"]
    queries = metadata.payload["queries"]
    bitstring = metadata.payload["bits"]

    def hashes(x):
        return [(a[j] * x + b[j]) % m for j in range(k)]

    bits = [int(ch) for ch in bitstring]
    abs_idx = [i for i, q in enumerate(queries) if any(bits[p] == 0 for p in hashes(q))]
    return sorted(abs_idx)


def test_gold_scores_1():
    random.seed(1)
    for level in [0, 1, 2, 3, 4, 5]:
        t = _new_task(level)
        for _ in range(20):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_gold_matches_independent_compute():
    random.seed(2)
    for level in [0, 3, 6]:
        t = _new_task(level)
        for _ in range(20):
            e = t.generate_example()
            assert _compute_absent(e.metadata) == [int(x) for x in e.answer.split()]


def test_absent_is_nontrivial():
    random.seed(3)
    for level in range(7):
        t = _new_task(level)
        for _ in range(20):
            e = t.generate_example()
            lst = e.answer.split()
            assert 1 <= len(lst) <= len(e.metadata.payload["queries"]) - 1


def test_bad_answers():
    random.seed(4)
    t = _new_task(3)
    e = t.generate_example()
    assert t.score_answer("", e) < 1.0
    assert t.score_answer("garbage nonsense", e) < 1.0
    assert t.score_answer("-3", e) < 1.0
    assert t.score_answer("999999", e) < 1.0


def test_wrong_subset():
    random.seed(5)
    t = _new_task(3)
    for _ in range(30):
        e = t.generate_example()
        gold = [int(x) for x in e.answer.split()]
        if len(gold) > 1:
            wrong = " ".join(str(i) for i in gold[:-1])
        else:
            wrong = " ".join(str(e.metadata.payload["m"] + i) for i in gold)
        if _parse(wrong) is not None:
            assert t.score_answer(wrong, e) < 1.0


def _parse(a):
    return mod._parse_indices(a)


def test_difficulty_updates():
    t = _new_task(0)
    base = (t.config.m_bits, t.config.k_hashes, t.config.n_items, t.config.t_queries)
    t = _new_task(3)
    high = (t.config.m_bits, t.config.k_hashes, t.config.n_items, t.config.t_queries)
    assert high[0] > base[0]
    assert high[3] > base[3]


def test_levels_generate():
    for level in range(7):
        t = _new_task(level)
        for _ in range(5):
            e = t.generate_example()
            assert e.answer is not None
