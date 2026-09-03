import math
import random

from reasoning_core.tasks.generated.wave3.s32_lexicographic_ranking.lexicographic_ranking import (
    LexicographicRanking,
    _counts,
    _total_arrangements,
    rank_multiset,
    word_at_rank,
)


def test_roundtrip_rank():
    for word in ["abc", "aab", "baa", "cba", "aabb", "abca", "aaaaa"]:
        total = _total_arrangements(word)
        for rank in range(1, total + 1):
            w = word_at_rank(rank, _counts(word))
            assert rank_multiset(w) == rank


def test_rank_bruteforce():
    def brute_rank(w):
        letters = sorted(w)
        res = []
        def rec(pref, rem):
            if not rem:
                res.append(pref)
                return
            seen = set()
            for ch in sorted(rem):
                if ch in seen:
                    continue
                seen.add(ch)
                idx = rem.index(ch)
                rec(pref + ch, rem[:idx] + rem[idx + 1:])
        rec("", w)
        return res.index(w) + 1

    random.seed(0)
    for _ in range(200):
        n = random.randint(3, 6)
        w = "".join(random.choice("abcd") for _ in range(n))
        assert rank_multiset(w) == brute_rank(w)


def test_bounds():
    for _ in range(50):
        n = random.randint(3, 7)
        w = "".join(random.choice("abc") for _ in range(n))
        total = _total_arrangements(w)
        r = random.randint(1, total)
        t = word_at_rank(r, _counts(w))
        assert rank_multiset(t) == r
        assert 1 <= r <= total


def test_task_example():
    random.seed(7)
    task = LexicographicRanking()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("zz", ex) == 0.0


def test_difficulty():
    task = LexicographicRanking()
    task.config.set_level(5)
    assert task.config.min_len > 3


def test_answer_space_wide():
    random.seed(11)
    task = LexicographicRanking()
    answers = set()
    for _ in range(100):
        ex = task.generate_example()
        answers.add(ex.answer)
    assert len(answers) > 20
