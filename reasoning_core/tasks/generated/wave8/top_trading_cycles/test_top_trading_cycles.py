import random

from reasoning_core.tasks.generated.wave8.top_trading_cycles.top_trading_cycles import (
    TopTradingCycles,
    TopTradingCyclesConfig,
    _run_ttc,
    _letter,
)


def test_gold_scores_one():
    random.seed(1)
    task = TopTradingCycles()
    for _ in range(40):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_wrong_answer_scores_zero():
    random.seed(2)
    task = TopTradingCycles()
    for _ in range(40):
        e = task.generate_example()
        for cand in ("A", "B", "C", "D", "E", "Z"):
            if cand != e.answer:
                assert task.score_answer(cand, e) == 0.0
                break


def test_junk_scores_zero():
    random.seed(3)
    task = TopTradingCycles()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("not an item", e) == 0.0


def test_matching_is_bijection():
    random.seed(4)
    for n in range(2, 12):
        prefs = [tuple(random.sample(range(n), n)) for _ in range(n)]
        match = _run_ttc(n, prefs)
        assert sorted(match) == list(range(n))
        assert sorted(match.values()) == list(range(n))


def test_difficulty_changes_config():
    cfg = TopTradingCyclesConfig()
    base = cfg.n_agents
    cfg.set_level(4)
    assert cfg.n_agents != base or cfg.n_agents > base
    assert cfg.n_agents >= base


def test_levels_produce_examples():
    task = TopTradingCycles()
    for level in (0, 1, 2, 3, 4, 5, 6):
        random.seed(level)
        e = task.generate_example(level=level)
        assert isinstance(e.answer, str) and e.answer
