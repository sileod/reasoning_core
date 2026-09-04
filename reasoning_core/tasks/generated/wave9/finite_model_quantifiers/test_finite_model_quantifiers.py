from reasoning_core.tasks.generated.wave9.finite_model_quantifiers.finite_model_quantifiers import (
    FiniteModelQuantifiers,
)


def _check_min_witness(ex):
    n = ex.metadata["n"]
    f = ex.metadata["f"]
    m = ex.metadata["m"]
    rel = {}
    recompute = FiniteModelQuantifiers().config_cls()
    # recompute relations via the same helper is internal; instead brute verify
    # using the payload-printed relation table.
    import ast

    payload = ex.metadata["payload"]
    rel2 = {x: [tuple(p) for p in payload["relation"][x]] for x in range(n)}
    found = None
    for x in range(n):
        if all(any((y, z) in rel2[x] and z != f[y] for z in range(n)) for y in range(n)):
            found = x
            break
    assert found == m, (found, m)
    assert ex.answer == str(m)
    assert int(ex.answer) in range(n)


def test_gold_scores_one():
    t = FiniteModelQuantifiers()
    for _ in range(50):
        ex = t.generate_example()
        _check_min_witness(ex)
        assert t.score_answer(ex.answer, ex) == 1.0


def test_wrong_answers_score_low():
    t = FiniteModelQuantifiers()
    for _ in range(50):
        ex = t.generate_example()
        n = ex.metadata["n"]
        for bad in [-1, n, n + 1, 999]:
            assert t.score_answer(str(bad), ex) < 1.0


def test_difficulty_changes():
    c = FiniteModelQuantifiers().config_cls()
    c.set_level(0)
    n0 = int(c.n)
    c.set_level(6)
    n6 = int(c.n)
    assert n6 > n0
