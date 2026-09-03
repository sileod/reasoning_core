from reasoning_core.tasks.generated.wave4.s39_multiplicative_order.s39_multiplicative_order import (
    _bsgs,
    _is_primitive_root,
    _multiplicative_order,
    MultiplicativeOrder,
)


def test_order_mode_gold():
    task = MultiplicativeOrder()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_order_computation():
    assert _multiplicative_order(2, 7) == 3
    assert _multiplicative_order(3, 7) == 6
    assert _multiplicative_order(1, 13) == 1


def test_bsgs():
    for p in (7, 13, 17, 19, 101):
        for g in range(2, p):
            if not _is_primitive_root(g, p):
                continue
            for h in range(1, p):
                k = _bsgs(g, h, p, p - 1)
                assert k is not None
                assert pow(g, k, p) == h


def test_answer_domain():
    task = MultiplicativeOrder()
    for _ in range(40):
        e = task.generate_example()
        if e.answer == "none":
            continue
        k = int(e.answer)
        m = e.metadata
        if m.payload["question"] == "order":
            assert k >= 1
            assert k % (m.payload["prime"] - 1) == 0 or (m.payload["prime"] - 1) % k == 0
        else:
            assert k >= 0
            assert pow(m.payload["base"], k, m.payload["prime"]) == m.payload["target"]
