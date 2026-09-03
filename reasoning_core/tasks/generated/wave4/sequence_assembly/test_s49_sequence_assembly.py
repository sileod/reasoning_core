import random
import pytest
from collections import Counter

from reasoning_core.tasks.generated.wave4.s49_sequence_assembly.s49_sequence_assembly import (
    SequenceAssembly, ArrayConfig, build_graph, count_eulerian_trails,
)


def test_gold_scoring():
    random.seed(3175732428)
    task = SequenceAssembly()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_wrong_answer_fails():
    random.seed(3175732428)
    task = SequenceAssembly()
    for _ in range(20):
        e = task.generate_example()
        if e.answer != "ambiguous":
            assert task.score_answer("ZZZZZ", e) == 0.0


def test_ambiguous_gold_accepts_only_ambiguous():
    random.seed(3175732428)
    task = SequenceAssembly()
    found = 0
    for _ in range(200):
        e = task.generate_example()
        if e.answer == "ambiguous":
            found += 1
            assert task.score_answer("ambiguous", e) == 1.0
            assert task.score_answer("ACGT", e) == 0.0
            if found >= 3:
                break
    assert found >= 3


def test_fragments_exact_multiset():
    random.seed(3175732428)
    task = SequenceAssembly()
    for _ in range(50):
        e = task.generate_example()
        s = e.answer
        if s == "ambiguous":
            continue
        frags = e.metadata.payload["fragments"]
        k = e.metadata.k
        gen = Counter(s[i:i + k] for i in range(len(s) - k + 1))
        assert Counter(frags) == gen


def test_difficulty_set_level():
    cfg = ArrayConfig()
    base = cfg.length
    cfg.set_level(6)
    assert cfg.length >= base


def test_scores_domain():
    random.seed(3175732428)
    task = SequenceAssembly()
    for _ in range(5):
        e = task.generate_example()
        assert task.score_answer("", e) == 0.0
