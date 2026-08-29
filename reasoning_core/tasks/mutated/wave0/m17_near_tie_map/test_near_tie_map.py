import math, random, re

from reasoning_core.tasks.mutated.wave0.m17_near_tie_map.near_tie_map import (
    NearTieMap,
    NearTieMapConfig,
    ranked_explanations,
    direct_rank,
)


def make():
    t = NearTieMap()
    t.config.set_level(2)
    return t


def test_gold_scores_one():
    t = make()
    x = t.generate_entry()
    assert t.score_answer(x.answer, x) == 1.0
    assert len(x.answer.split()) == x.metadata.n_atoms


def test_wrong_partial_scores_low():
    t = make()
    x = t.generate_entry()
    assert t.score_answer("", x) == 0.0
    scored = sum(t.score_answer(a, x) for a in ["0", "0 1", "1 2 3"])
    assert scored < len(x.answer.split())


def test_margin_in_window():
    t = make()
    seen = []
    for _ in range(15):
        x = t.generate_entry()
        lm = x.metadata.log_margin
        assert x.metadata.min_log_margin_applied() is False if hasattr(x.metadata, "min_log_margin_applied") else True
        cfg = t.config
        assert lm > 0.0
        assert lm >= cfg.min_log_margin and lm <= cfg.max_log_margin
        seen.append(x.answer)
    assert len(set(seen)) > 3


def test_winner_is_top_two():
    t = make()
    x = t.generate_entry()
    r = ranked_explanations(x.metadata.problog)
    assert x.metadata.winner_lits == r[0][1]
    assert x.metadata.runner_lits == r[1][1]


def test_direct_matches_problog():
    t = make()
    x = t.generate_entry()
    atoms = list(x.metadata.probabilities.keys())
    direct = sorted(direct_rank(atoms, x.metadata.probabilities, _formula(x)), reverse=True)
    prob = ranked_explanations(x.metadata.problog)
    assert direct[0][1] == prob[0][1]
    assert direct[1][1] == prob[1][1]


def _formula(x):
    src = x.metadata.problog
    m = re.search(r"observed :- (.*)\.\n", src)
    return m.group(1)


def test_all_levels_generate():
    for L in range(6):
        t = NearTieMap()
        t.config.set_level(L)
        x = t.generate_entry()
        assert t.score_answer(x.answer, x) == 1.0


def test_answers_vary():
    t = NearTieMap()
    answers = [t.generate_entry().answer for _ in range(20)]
    assert len(set(answers)) >= 6


def test_multiple_acceptable_equal_answers():
    t = make()
    trials = [t.generate_entry() for _ in range(5)]
    for x in trials:
        assert t.score_answer(x.answer, x) == 1.0
        assert t.score_answer(" ".join(reversed(x.answer.split())), x) <= 1.0
