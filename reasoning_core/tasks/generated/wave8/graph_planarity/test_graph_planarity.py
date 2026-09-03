import random

from reasoning_core.tasks.generated.wave8.graph_planarity.graph_planarity import (
    GraphPlanarity,
    _genus_complete,
    _genus_complete_bipartite,
)


def test_genus_complete_k4_to_k10():
    known = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 2, 9: 3, 10: 4}
    for n, g in known.items():
        assert _genus_complete(n) == g


def test_genus_complete_bipartite_k33():
    assert _genus_complete_bipartite(3, 3) == 1
    assert _genus_complete_bipartite(4, 4) == 1
    assert _genus_complete_bipartite(4, 5) == 2


def test_planar_is_zero():
    assert _genus_complete(4) == 0
    assert _genus_complete_bipartite(2, 9) == 0


def test_generate_and_score():
    random.seed(3982003255)
    t = GraphPlanarity()
    for L in (0, 2, 5):
        t.config.set_level(L)
        for _ in range(5):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0
            assert t.score_answer("", e) == 0.0
            assert t.score_answer("not a number", e) == 0.0
            g = int(e.answer)
            assert g >= 0


def test_answer_variety():
    random.seed(3982003255)
    t = GraphPlanarity()
    t.config.set_level(0)
    answers = {t.generate_example().answer for _ in range(60)}
    assert len(answers) >= 3
