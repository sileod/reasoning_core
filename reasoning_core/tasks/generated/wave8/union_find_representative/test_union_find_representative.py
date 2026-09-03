from reasoning_core.tasks.generated.wave8.union_find_representative.union_find_representative import (
    UnionFindRepresentative,
    UnionFindConfig,
    run_ops,
    find_plain,
    q_rep,
)


def test_gold_scores():
    task = UnionFindRepresentative()
    for _ in range(20):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0
        assert task.score_answer("", x) < 1.0
        assert task.score_answer("[-1]", x) < 1.0


def test_answer_gold():
    import random
    random.seed(1)
    task = UnionFindRepresentative()
    for _ in range(20):
        x = task.generate_example()
        ops = x.metadata.ops
        parent, rank, last_q = run_ops(ops, x.metadata.n, x.metadata.tie_smaller)
        assert int(x.answer) == q_rep(parent, rank, x.metadata.query)
        assert find_plain(parent, int(x.answer)) == int(x.answer)


def test_difficulty():
    cfg = UnionFindConfig()
    cfg.set_level(0)
    l0 = (cfg.n_nodes, cfg.n_ops)
    cfg = UnionFindConfig()
    cfg.set_level(5)
    l5 = (cfg.n_nodes, cfg.n_ops)
    assert l5[0] > l0[0]
    assert l5[1] > l0[1]


def test_all_levels_generate():
    task = UnionFindRepresentative()
    for lvl in range(0, 7):
        task.config.set_level(lvl)
        for _ in range(5):
            x = task.generate_example()
            assert task.score_answer(x.answer, x) == 1.0
