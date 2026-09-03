import random

from reasoning_core.tasks.generated.wave8.trie_unique_prefix.trie_unique_prefix import (
    TrieUniquePrefix,
    _unique_prefix,
)


def _solve(words, target):
    return _unique_prefix(list(words), target)


def test_roundtrip_scores_1():
    random.seed(123)
    task = TrieUniquePrefix()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert ex.metadata is not None
            assert isinstance(ex.answer, str)
            assert task.score_answer(ex.answer, ex) == 1.0


def test_answer_is_shortest_unique_prefix():
    random.seed(456)
    task = TrieUniquePrefix()
    for _ in range(60):
        ex = task.generate_example()
        words = list(ex.metadata["words"])
        target = ex.metadata["target"]
        ans = ex.answer
        # ans is a prefix of target
        assert target.startswith(ans)
        # ans matches exactly one word (the target)
        hits = [w.startswith(ans) for w in words]
        assert sum(hits) == 1
        assert hits[words.index(target)]
        # shorten one char -> not unique anymore (unless ans is length 1, still unique)
        if len(ans) > 1:
            shorter = ans[:-1]
            assert sum(w.startswith(shorter) for w in words) > 1
        else:
            assert sum(w.startswith(ans[:0]) for w in words) >= 1


def test_wrong_answers_score_0():
    random.seed(789)
    task = TrieUniquePrefix()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer("", ex) == 0.0
        wrong = ex.answer + "x"
        assert task.score_answer(wrong, ex) == 0.0
