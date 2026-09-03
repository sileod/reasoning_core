import random
from reasoning_core.tasks.generated.wave3.s34_linear_recurrences.s34_linear_recurrences import (
    LinearRecurrence,
    LinearRecurrenceConfig,
    _recurrence_term,
    _generate,
)


def test_gold_scores_one():
    random.seed(1)
    task = LinearRecurrence()
    for _ in range(20):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_garbage_scores_zero():
    task = LinearRecurrence()
    x = task.generate_example()
    assert task.score_answer('', x) == 0.0
    assert task.score_answer('junk', x) == 0.0


def test_matches_brute_force():
    random.seed(2)
    task = LinearRecurrence()
    for _ in range(50):
        x = task.generate_example()
        coeffs = x.metadata.coeffs
        inits = x.metadata.inits
        index = x.metadata.index
        order = x.metadata.order
        state = list(reversed(inits))
        if index < order:
            expected = inits[index]
        else:
            for _ in range(index - order + 1):
                nxt = sum(c * s for c, s in zip(coeffs, state))
                state = [nxt] + state[:-1]
            expected = state[0]
        assert int(x.answer) == expected


def test_difficulty_changes():
    cfg = LinearRecurrenceConfig()
    idx1 = cfg.set_level(1).index
    cfg2 = LinearRecurrenceConfig()
    idx2 = cfg2.set_level(3).index
    assert idx2 > idx1


def test_answer_not_in_prompt():
    random.seed(3)
    task = LinearRecurrence()
    for _ in range(30):
        x = task.generate_example()
        ans = int(x.answer)
        assert ans not in x.metadata.coeffs
        assert ans not in x.metadata.inits
        assert ans != x.metadata.index


def test_mat_pow_matches_brute_mod():
    import random as r
    r.seed(4)
    for _ in range(30):
        coeffs = [r.randint(-5, 5) for _ in range(3)]
        inits = [r.randint(-5, 5) for _ in range(3)]
        idx = r.randint(0, 40)
        got = _recurrence_term(coeffs, inits, idx)
        state = list(reversed(inits))
        if idx < 3:
            assert got == inits[idx]
        else:
            for _ in range(idx - 3 + 1):
                nxt = (coeffs[0] * state[0] + coeffs[1] * state[1]
                       + coeffs[2] * state[2])
                state = [nxt] + state[:-1]
            assert got == state[0]
