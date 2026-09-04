import random

from reasoning_core.tasks.generated.wave9.hidden_state_filtering.hidden_state_filtering import (
    HiddenStateFiltering,
    _filter,
    _filter2,
)


def test_gold_scores_one():
    task = HiddenStateFiltering()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_all_levels_generate():
    task = HiddenStateFiltering()
    for level in (0, 1, 2, 3, 4, 5, 6):
        task.config.set_level(level)
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0
        assert 0.0 <= float(x.answer) <= 1.0


def test_junk_and_empty_rejected():
    task = HiddenStateFiltering()
    for _ in range(20):
        x = task.generate_example()
        assert task.score_answer("", x) < 1.0
        assert task.score_answer("reajrjrje9595!", x) < 1.0
        assert task.score_answer("abc", x) < 1.0


def test_area_beyond_tolerance_rejected():
    task = HiddenStateFiltering()
    for _ in range(20):
        x = task.generate_example()
        gold = float(x.answer)
        assert task.score_answer(f"{gold + 0.05:.3f}", x) < 1.0


def test_forward_crosscheck():
    random.seed(7)
    from reasoning_core.tasks.generated.wave9.hidden_state_filtering.hidden_state_filtering import (
        _row_stochastic,
    )
    for _ in range(50):
        K = random.randint(2, 6)
        M = random.randint(2, 6)
        T = random.randint(2, 20)
        init = _row_stochastic(random, K)
        trans = [_row_stochastic(random, K) for _ in range(K)]
        emit = [_row_stochastic(random, M) for _ in range(K)]
        obs = [random.randrange(M) for _ in range(T)]
        b1 = _filter(init, trans, emit, obs)
        b2 = _filter2(init, trans, emit, obs)
        for t in range(T):
            for s in range(K):
                assert abs(b1[t][s] - b2[t][s]) < 1e-9
            assert abs(sum(b1[t]) - 1.0) < 1e-9
            assert all(0.0 <= v <= 1.0 for v in b1[t])


def test_balanced_distinct_answers():
    task = HiddenStateFiltering()
    answers = {task.generate_example().answer for _ in range(40)}
    assert len(answers) >= 10
