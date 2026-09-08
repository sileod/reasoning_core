import re

import pytest
from sympy.geometry import Line, Point

from easydict import EasyDict as edict

from reasoning_core.tasks import math_geometry
from reasoning_core.tasks.math_geometry import (
    PlanarGeometryRelationsConfig,
    angle_type,
    make_scene,
    query_line_intersection,
    render,
    render_text,
    triangle_position,
)


def test_render_text_replaces_compact_point_refs():
    labels = {"p1": "E", "p2": "F", "p6": "Q"}
    assert render_text("triangle p1p6p2", labels) == "triangle EQF"


def test_triangle_vertex_is_boundary():
    A, B, C = Point(0, 0), Point(2, 0), Point(0, 2)
    assert triangle_position(A, B, C, A) == "boundary"


def test_degenerate_angles_are_not_classified():
    assert angle_type(Point(1, 0), Point(0, 0), Point(2, 0)) is None
    assert angle_type(Point(1, 0), Point(0, 0), Point(-2, 0)) is None


def test_none_line_intersection_is_distinct_parallel(monkeypatch):
    cfg = PlanarGeometryRelationsConfig(n_constructed_points=0)
    scene = make_scene(cfg)
    choose = math_geometry.random.choice
    first = True

    def choose_none(options):
        nonlocal first
        if first:
            first = False
            return "none"
        return choose(options)

    monkeypatch.setattr(math_geometry.random, "choice", choose_none)
    query = query_line_intersection(scene, cfg)
    assert query.answer == "none"
    points = dict(scene["points"], **{i: P for i, P, _, _ in query.additions})
    a, b, c, d = re.findall(r"p\d+", query.question)
    assert Line(points[a], points[b]).intersection(Line(points[c], points[d])) == []
    assert math_geometry.cross(points[a], points[b], points[c]) != 0


@pytest.mark.parametrize(
    ("query_fn", "answers", "n_additions"),
    [
        (math_geometry.query_orientation, ["left", "right", "on"], 2),
        (math_geometry.query_collinear, ["yes", "no"], 2),
        (math_geometry.query_line_relation, ["parallel", "perpendicular", "neither"], 3),
        (math_geometry.query_line_intersection, ["point", "none"], 3),
        (math_geometry.query_segment_intersection, ["yes", "no"], 3),
        (math_geometry.query_between, ["yes", "no"], 2),
        (math_geometry.query_angle_type, ["acute", "right", "obtuse"], 2),
        (math_geometry.query_inside_triangle, ["inside", "outside", "boundary"], 3),
        (math_geometry.query_closer, ["first", "second", "tie"], 2),
    ],
)
def test_answer_classes_add_the_same_number_of_local_points(monkeypatch, query_fn, answers, n_additions):
    cfg = PlanarGeometryRelationsConfig(n_constructed_points=0)
    scene = make_scene(cfg)
    choose = math_geometry.random.choice

    for wanted in answers:
        first = True

        def choose_answer(options):
            nonlocal first
            if first:
                first = False
                return wanted
            return choose(options)

        monkeypatch.setattr(math_geometry.random, "choice", choose_answer)
        query = query_fn(scene, cfg)
        assert query is not None
        assert len(query.additions) == n_additions
        if wanted in {"yes", "no"}:
            assert query.answer == wanted.capitalize()
            assert "Yes or No" in query.instruction
        monkeypatch.setattr(math_geometry.random, "choice", choose)


def test_render_separates_coordinate_and_construction_modes(monkeypatch):
    scene = {
        "points": {"p0": Point(0, 0), "p1": Point(2, 0), "p2": Point(1, 0)},
        "definitions": {"p2": "the midpoint of p0 and p1"},
        "depth": {"p0": 0, "p1": 0, "p2": 1},
    }
    query = edict(
        additions=[],
        kind="choice",
        answer="Yes",
        question="Is point p2 on segment p0p1?",
        instruction="Answer is either Yes or No.",
        type="between",
        balance="between:yes",
    )
    monkeypatch.setattr(math_geometry.random, "sample", lambda population, count: list(population))

    coordinate = render(scene, query, construction_execution=False)
    construction = render(scene, query, construction_execution=True)

    assert coordinate.points == {"A": "(0, 0)", "B": "(2, 0)", "C": "(1, 0)"}
    assert coordinate.definitions == []
    assert construction.points == {"A": "(0, 0)", "B": "(2, 0)"}
    assert construction.definitions == ["C is the midpoint of A and B."]
