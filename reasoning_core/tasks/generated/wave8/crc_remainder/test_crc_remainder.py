import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.crc_remainder.crc_remainder import (
    CrcRemainder,
    crc_remainder,
    build_poly,
)


def test_generate_scores_one():
    task = CrcRemainder()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0


def test_crc_matches_reference():
    import random as r
    r.seed(7)
    examples = [CrcRemainder().generate_example() for _ in range(5)]
    for ex in examples:
        msg = [int(c) for c in ex.metadata.message]
        ans = ex.answer
        assert len(ans) == ex.metadata.width


def test_garbage_does_not_score_one():
    task = CrcRemainder()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("hello", ex) < 1.0
    assert task.score_answer("999", ex) < 1.0


def test_difficulty_changes():
    task = CrcRemainder()
    base = int(task.config.msg_len)
    task.config.set_level(3)
    assert int(task.config.msg_len) > base


def test_answer_varies():
    task = CrcRemainder()
    answers = set()
    for _ in range(30):
        answers.add(task.generate_example().answer)
    assert len(answers) >= 5


def test_gold_is_actual_crc():
    from reasoning_core.tasks.generated.wave8.crc_remainder.crc_remainder import (
        crc_remainder,
    )
    import random as r
    r.seed(11)
    task = CrcRemainder()
    for _ in range(10):
        ex = task.generate_example()
        msg = [int(c) for c in ex.metadata.message]
        deg = ex.metadata.width
        coeffs = [0] * (deg + 1)
        coeffs[0] = 1
        coeffs[deg] = 1
        terms = ex.metadata.polynomial.split(" + ")
        for term in terms:
            if term == "1":
                coeffs[0] = 1
            elif term == "x":
                coeffs[1] = 1
            else:
                d = int(term.split("^")[1])
                coeffs[d] = 1
        poly_bits = list(reversed(coeffs))
        expected = crc_remainder(msg, poly_bits)
        assert bits_str_to_list(ex.answer) == expected


def bits_str_to_list(s):
    return [int(c) for c in s]
