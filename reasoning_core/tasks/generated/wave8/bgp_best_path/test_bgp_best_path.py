import math
import random

from reasoning_core.template import Entry, edict
from reasoning_core.tasks.generated.wave8.bgp_best_path.bgp_best_path import (
    BgpBestPath,
    ORIGIN_ORDER,
    _select_best,
)


def _make_entry(routes):
    winner = _select_best(routes)
    payload = "\n".join(
        "%s local-pref %d as-path-length %d origin %s med %d neighbor %s" % (
            r["name"], r["local_pref"], r["as_path"], r["origin"], r["med"], r["neighbor"])
        for r in routes
    )
    return Entry(metadata=edict({"routes": routes, "winner": winner["name"],
                                 "payload": {"routes": payload}}), answer=winner["name"])


def test_gold_scores_one():
    task = BgpBestPath()
    task.generate_entry()
    for _ in range(40):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    task = BgpBestPath()
    e = task.generate_entry()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0


def test_select_best_respects_rules():
    routes = [
        {"name": "A", "local_pref": 100, "as_path": 3, "origin": "IGP", "med": 30, "neighbor": "10.1.1.1"},
        {"name": "B", "local_pref": 200, "as_path": 5, "origin": "IGP", "med": 10, "neighbor": "10.2.2.1"},
        {"name": "C", "local_pref": 200, "as_path": 4, "origin": "IGP", "med": 5, "neighbor": "10.3.3.1"},
    ]
    assert _select_best(routes)["name"] == "C"


def test_local_pref_higher_wins():
    routes = [
        {"name": "A", "local_pref": 100, "as_path": 1, "origin": "IGP", "med": 0, "neighbor": "10.1.1.1"},
        {"name": "B", "local_pref": 150, "as_path": 5, "origin": "incomplete", "med": 99, "neighbor": "10.9.9.1"},
    ]
    assert _select_best(routes)["name"] == "B"


def test_tiebreak_neighbor_ip():
    routes = [
        {"name": "A", "local_pref": 100, "as_path": 2, "origin": "IGP", "med": 10, "neighbor": "10.2.2.1"},
        {"name": "B", "local_pref": 100, "as_path": 2, "origin": "IGP", "med": 10, "neighbor": "10.1.1.1"},
    ]
    assert _select_best(routes)["name"] == "B"


def test_difficulty_changes_config():
    task = BgpBestPath()
    task.config.set_level(0)
    n0 = task.config.n_routes
    task.config.set_level(5)
    n5 = task.config.n_routes
    assert n5 > n0


def test_winner_distribution_wide():
    random.seed(1)
    task = BgpBestPath()
    winners = set()
    for _ in range(120):
        winners.add(task.generate_entry().answer)
    assert len(winners) >= 3
