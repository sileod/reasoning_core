import random

from reasoning_core.tasks.tracking import ReferenceTracking


def test_swap_moves_exchange_single_occupants():
    placement = {"b1": "x1", "b2": "x2", "b3": "x3"}

    moves, resolved, affected = ReferenceTracking()._do_moves(
        placement, list(placement), ["x1", "x2", "x3"],
        n_steps=1, bulk_p=0.0, pronoun_p=0.0, swap_p=1.0,
    )

    assert moves == resolved
    assert moves[0].startswith("Swap the balls in ")
    assert sorted(placement.values()) == ["x1", "x2", "x3"]
    assert placement != {"b1": "x1", "b2": "x2", "b3": "x3"}
    assert len(affected) == 2
    assert affected < set(placement)


def test_tracking_targets_prefer_affected_balls_but_keep_controls():
    state = random.getstate()
    try:
        random.seed(0)
        task = ReferenceTracking()
        balls = ["b1", "b2", "b3", "b4"]
        affected = {"b1", "b2"}
        targets = [task._tracking_target(balls, affected, 0.8) for _ in range(1000)]
        affected_rate = sum(target in affected for target in targets) / len(targets)

        assert 0.75 < affected_rate < 0.85
        assert set(targets) == set(balls)
    finally:
        random.setstate(state)
