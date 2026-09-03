import re

from reasoning_core.tasks.generated.wave8.clock_page_replacement.clock_page_replacement import (
    ClockPageReplacement,
    clock_victim,
)


def _run(level, seed):
    task = ClockPageReplacement()
    task.config.set_level(level)
    e = task.generate_example(seed=seed)
    return e


def test_gold_scores_one():
    for level in (0, 1, 2, 3, 4, 5, 6):
        e = _run(level, seed=100 + level)
        assert task_roundtrip(level, e), f"gold did not score 1 at level {level}"


def task_roundtrip(level, e):
    task = ClockPageReplacement()
    task.config.set_level(level)
    return task.score_answer(e.answer, e) == 1.0


def test_answer_is_int():
    for level in (0, 2, 5):
        e = _run(level, seed=200 + level)
        assert re.fullmatch(r"\d+", e.answer)


def test_victim_matches_prompt_algorithm():
    task = ClockPageReplacement()
    task.config.set_level(3)
    for _ in range(50):
        e = task.generate_example()
        n = e.metadata["n_frames"]
        pages = e.metadata["pages"]
        bits = e.metadata["bits"]
        hand = e.metadata["hand"]
        victim = clock_victim(pages, bits, hand)
        assert int(e.answer) == victim
        assert 0 <= victim < n


def test_difficulty_changes_size():
    low = ClockPageReplacement()
    low.config.set_level(0)
    e0 = low.generate_example()
    high = ClockPageReplacement()
    high.config.set_level(6)
    e6 = high.generate_example()
    assert e6.metadata["n_frames"] > e0.metadata["n_frames"]


def test_empty_and_junk_not_full_credit():
    task = ClockPageReplacement()
    task.config.set_level(0)
    e = task.generate_example()
    assert task.score_answer("", e) < 1.0
    assert task.score_answer("banana", e) < 1.0


def test_clock_always_finds_zero_bit():
    task = ClockPageReplacement()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(200):
            e = task.generate_example()
            assert any(b == 0 for b in e.metadata["bits"]), f"no 0-bit at level {level}"
