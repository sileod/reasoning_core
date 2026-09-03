import re

from reasoning_core.tasks.generated.wave8.pattern_match_exhaustiveness.pattern_match_exhaustiveness import (
    PatternMatchExhaustiveness,
)


def _compute_uncovered(names, clauses):
    covered = set()
    for cl in clauses:
        for name in re.findall(r"c\d+", cl):
            covered.add(name)
    return len([n for n in names if n not in covered])


def test_gold_scoring():
    task = PatternMatchExhaustiveness()
    for _ in range(200):
        entry = task.generate_example()
        gold = entry.answer
        assert task.score_answer(gold, entry) == 1.0
        assert entry.answer.lstrip("-").isdigit()
        n = int(entry.answer)
        cnames = set(entry.metadata.names)
        assert 0 <= n <= len(cnames)
        assert _compute_uncovered(cnames, entry.metadata.clauses) == n


def test_junk_and_empty_not_full():
    task = PatternMatchExhaustiveness()
    entry = task.generate_example()
    assert task.score_answer("", entry) < 1.0
    assert task.score_answer("garbage", entry) < 1.0
    assert task.score_answer("", entry) == 0.0


def test_answers_vary():
    task = PatternMatchExhaustiveness()
    answers = {task.generate_example().answer for _ in range(300)}
    assert len(answers) >= 4


def test_difficulty_changes():
    cfg = PatternMatchExhaustiveness.config_cls()
    base = (cfg.k_min, cfg.k_max)
    cfg.set_level(6)
    assert (cfg.k_min, cfg.k_max) != base
