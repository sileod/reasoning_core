import random

from reasoning_core.tasks.generated.wave2.s25_lattice_polygons.s25_lattice_polygons import (
    LatticePolygons,
    LatticePolygonsConfig,
    _is_simple,
)


def _render(task, ex):
    return task.render_prompt(ex.metadata)


def test_template_config():
    cfg = LatticePolygonsConfig()
    cfg.set_level(5)
    assert cfg.n_verts > LatticePolygonsConfig().n_verts
    assert cfg.coord_range > LatticePolygonsConfig().coord_range


def test_simple_polygons():
    random.seed(3944590077)
    task = LatticePolygons()
    for level in [0, 3, 6]:
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            v = ex.metadata.vertices
            assert _is_simple(v)
            assert len(set(map(tuple, v))) == len(v)


def test_pick_theorem_and_scoring():
    random.seed(3944590077)
    task = LatticePolygons()
    for level in [0, 2, 5]:
        task.config.set_level(level)
        for _ in range(10):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0
            k = ex.metadata.kind
            interior = ex.metadata.interior
            boundary = ex.metadata.boundary
            area2 = ex.metadata.area2
            if k == 'interior':
                assert int(ex.answer) == interior and interior >= 0
            elif k == 'boundary':
                assert int(ex.answer) == boundary and boundary > 0
            else:
                assert int(ex.answer) == area2 and area2 > 0


def test_garbage_answer():
    random.seed(1)
    task = LatticePolygons()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("not a number", ex) == 0.0


def test_prompt_has_vertices():
    random.seed(2)
    task = LatticePolygons()
    ex = task.generate_example()
    p = task.render_prompt(ex.metadata)
    assert "integer" in p


def test_answer_variety():
    random.seed(3)
    task = LatticePolygons()
    answers = set()
    for _ in range(60):
        ex = task.generate_example()
        answers.add(int(ex.answer))
    assert len(answers) > 10
