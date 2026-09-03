from reasoning_core.tasks.generated.wave3.s31_grundy_values.grundy_values import (
    GrundyValues,
    GrundyValuesConfig,
    _grundy_table,
)


def test_config_difficulty_increases():
    cfg = GrundyValuesConfig()
    base = (cfg.n_heaps, cfg.max_size, cfg.n_distinct_rules)
    cfg.set_level(3)
    hi = (cfg.n_heaps, cfg.max_size, cfg.n_distinct_rules)
    assert hi[0] >= base[0] and hi[1] >= base[1] and hi[2] >= base[2]
    cfg.set_level(6)
    hi6 = (cfg.n_heaps, cfg.max_size, cfg.n_distinct_rules)
    assert hi6[0] >= hi[0] and hi6[1] >= hi[1] and hi6[2] >= hi[2]


def test_grundy_take_known_values():
    g = _grundy_table("take", 10)
    assert g[0] == 0
    assert g[1] == 1
    assert g[4] == 0  # periodic mod 4 for subtraction {1,2,3}


def test_grundy_square_known():
    g = _grundy_table("square", 10)
    assert g[0] == 0
    assert g[1] == 1
    assert g[2] == 0
    assert g[4] == 2


def test_grundy_split_small():
    g = _grundy_table("split", 6)
    assert g[2] == 0  # only split into two 1s = equal, not allowed -> no moves
    assert g[3] == 1  # 1|2


def test_whole_is_xor_of_parts():
    task = GrundyValues()
    for _ in range(30):
        ex = task.generate_example()
        parts = [int(x) for x in ex.answer.split()]
        per, whole = parts[:-1], parts[-1]
        x = 0
        for v in per:
            x ^= v
        assert x == whole


def test_generate_example_scores_one():
    for _ in range(20):
        task = GrundyValues()
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_wrong_answers_do_not_score_one():
    task = GrundyValues()
    for _ in range(30):
        ex = task.generate_example()
        parts = ex.answer.split()
        parts[0] = str((int(parts[0]) + 1) % 3)
        assert task.score_answer(" ".join(parts), ex) == 0.0
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("junk", ex) == 0.0
    assert task.score_answer("0 0", ex) == 0.0
