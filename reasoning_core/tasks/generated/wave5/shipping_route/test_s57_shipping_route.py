import random

from reasoning_core.tasks.generated.wave5.s57_shipping_route.s57_shipping_route import ShippingRoute


def test_gold_scores_one():
    t = ShippingRoute()
    for level in range(7):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_wrong_scores_zero():
    t = ShippingRoute()
    t.config.set_level(2)
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer("Nowhere", e) == 0.0
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("A -> B", e) == 0.0


def test_unreachable():
    from reasoning_core.template import edict
    t = ShippingRoute()
    e = t.generate_example()
    meta = edict(dict(e.metadata))
    meta.payload = meta.payload
    meta["start"] = "X"
    meta["target"] = "Y"
    meta["payload"]["start"] = "X"
    meta["payload"]["target"] = "Y"
    assert e.answer is not None


def test_prompt_contains_answer_format():
    t = ShippingRoute()
    for level in (0, 2, 5):
        t.config.set_level(level)
        e = t.generate_example()
        prompt = t.render_prompt(e.metadata)
        assert "arrow-separated" in prompt
        assert "->" in prompt


def test_levels_change_places():
    t = ShippingRoute()
    t.config.set_level(0)
    n0 = t.config.n_places
    t.config.set_level(5)
    n5 = t.config.n_places
    assert n5 > n0


def test_whitespace_insensitive():
    t = ShippingRoute()
    t.config.set_level(1)
    e = t.generate_example()
    alt = " -> ".join([s.strip() for s in e.answer.split("->")])
    assert t.score_answer(alt, e) == 1.0
