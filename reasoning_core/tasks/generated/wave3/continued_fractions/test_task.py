import random
from fractions import Fraction

from reasoning_core.template import edict
from reasoning_core.tasks.generated.wave3.s26_continued_fractions.task import (
    ContinuedFractions,
    ContinuedFractionsConfig,
)

SEED = 3369804028


def test_expansion_gold_and_score():
    random.seed(SEED)
    task = ContinuedFractions(config_cls=ContinuedFractionsConfig)
    for _ in range(50):
        entry = task.generate_example()
        assert isinstance(entry.metadata.payload, dict)
        if entry.metadata.kind == "expansion":
            coeffs = [int(x) for x in entry.answer.split(",")]
            value = Fraction(coeffs[-1])
            for c in reversed(coeffs[:-1]):
                value = c + 1 / value
            assert value > 0
        assert task.score_answer(entry.answer, entry) == 1.0


def test_score_rejects_junk():
    random.seed(SEED)
    task = ContinuedFractions(config_cls=ContinuedFractionsConfig)
    entries = [task.generate_example() for _ in range(20)]
    for entry in entries:
        assert task.score_answer("", entry) < 1.0
        assert task.score_answer("garbage", entry) < 1.0
        assert task.score_answer(None, entry) < 1.0


def test_convergent_correctness():
    from reasoning_core.tasks.generated.wave3.s26_continued_fractions.task import _convergent

    random.seed(SEED)
    task = ContinuedFractions(config_cls=ContinuedFractionsConfig)
    for _ in range(50):
        entry = task.generate_example()
        if entry.metadata.kind == "convergent":
            coeffs = list(entry.metadata.coeffs)
            k = int(entry.metadata.k)
            num, den = _convergent(coeffs, k)
            g = _gcd(num, den)
            assert "%d/%d" % (num // g, den // g) == entry.answer


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a
