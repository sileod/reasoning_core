import random

import pytest

from reasoning_core.tasks.generated.wave8.longest_prefix_route.longest_prefix_route import (
    LongestPrefixRoute,
    LongestPrefixRouteConfig,
)

MOD = "reasoning_core.tasks.generated.wave8.longest_prefix_route.longest_prefix_route"


def test_generate_scores_own_answer_all_levels():
    task = LongestPrefixRoute()
    for level in range(7):
        cfg = LongestPrefixRouteConfig()
        cfg.set_level(level)
        task.config = cfg
        ex = task.generate_entry()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_winning_hop_is_longest_match():
    task = LongestPrefixRoute()
    ex = task.generate_entry()
    dest = ex.metadata["destination"]
    routes = [(p, l, h) for (l, p, h) in ex.metadata["routes"]]
    matches = [h for (p, l, h) in routes if dest.startswith(p)]
    assert matches
    best = max(l for (p, l, h) in routes if dest.startswith(p))
    best_hops = [h for (p, l, h) in routes if dest.startswith(p) and l == best]
    assert len(set(best_hops)) == 1
    assert ex.metadata["winning_hop"] == best_hops[0]
    assert int(ex.answer) == best_hops[0]


def test_difficulty_changes():
    base = LongestPrefixRouteConfig()
    high = LongestPrefixRouteConfig()
    high.set_level(6)
    assert high.n_routes > base.n_routes
    assert high.width >= base.width


def test_junk_and_empty_score_zero():
    task = LongestPrefixRoute()
    ex = task.generate_entry()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("not a number", ex) == 0.0
    assert task.score_answer("1001", ex) in (0.0, 1.0)


def test_default_route_always_present():
    task = LongestPrefixRoute()
    for _ in range(20):
        ex = task.generate_entry()
        assert any(l == 0 for (l, p, h) in ex.metadata["routes"])


def test_distance_from_surface():
    task = LongestPrefixRoute()
    varied = set()
    for _ in range(30):
        ex = task.generate_entry()
        varied.add(ex.answer)
    assert len(varied) >= 5
