from fractions import Fraction

from reasoning_core.tasks.generated.wave3.s37_rational_plane_geometry.s37_rational_plane_geometry import (
    RationalPlaneGeometry,
)


def _make(level=1):
    t = RationalPlaneGeometry()
    t.config.set_level(level)
    return t


def test_gold_scores_one():
    t = _make()
    for _ in range(30):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    t = _make()
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("garbage!!", e) == 0.0
        assert t.score_answer(None, e) == 0.0


def test_area_strictly_positive():
    t = _make(3)
    for _ in range(20):
        e = t.generate_example()
        if e.metadata.kind == "area":
            assert Fraction(e.metadata.value) > 0


def test_intersection_verifiable():
    t = _make(2)
    for _ in range(20):
        e = t.generate_example()
        if e.metadata.kind == "intersection":
            p = e.metadata.payload
            fx, fy = map(lambda s: Fraction(s), e.answer.split(","))
            assert _on_line(fx, fy, p["point_a"], p["point_b"])
            assert _on_line(fx, fy, p["point_c"], p["point_d"])


def _on_line(fx, fy, p, q):
    return (q[0] - p[0]) * (fy - p[1]) - (q[1] - p[1]) * (fx - p[0]) == 0
