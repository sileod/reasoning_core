from reasoning_core.tasks.generated.wave8.quorum_intersection.quorum_intersection import (
    QuorumIntersection,
    QuorumIntersectionConfig,
    _guaranteed_intersection,
)


def test_gold_roundtrip():
    t = QuorumIntersection()
    for _ in range(200):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0
        n = e.metadata.replicas
        r = e.metadata.read_size
        w = e.metadata.write_size
        assert e.answer == str(max(0, r + w - n))


def test_formula():
    assert _guaranteed_intersection(10, 6, 7) == 3
    assert _guaranteed_intersection(10, 6, 4) == 0
    assert _guaranteed_intersection(10, 10, 10) == 10
    assert _guaranteed_intersection(5, 1, 1) == 0


def test_difficulty_changes():
    c = QuorumIntersectionConfig()
    c2 = QuorumIntersectionConfig()
    c2.set_level(4)
    assert c2.n_hi > c.n_hi


def test_junk_scoring():
    t = QuorumIntersection()
    e = t.generate_example()
    assert t.score_answer("", e) < 1.0
    assert t.score_answer("not a number", e) < 1.0
    assert t.score_answer("999999", e) < 1.0


def test_domain():
    t = QuorumIntersection()
    for _ in range(300):
        e = t.generate_example()
        n = e.metadata.replicas
        g = int(e.answer)
        assert 0 <= g <= min(e.metadata.read_size, e.metadata.write_size)
        assert n in range(5, 20)
