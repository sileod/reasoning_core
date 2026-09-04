import random

from reasoning_core.template import Entry, edict
from reasoning_core.tasks.generated.wave9.bayesian_conditioning.bayesian_conditioning import (
    BayesianConditioning,
    BayesianConfig,
)


def test_gold_scores_one():
    t = BayesianConditioning()
    ex = t.generate_example()
    assert t.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_zero():
    t = BayesianConditioning()
    ex = t.generate_example()
    assert t.score_answer("", ex) < 1.0
    assert t.score_answer("garbage", ex) < 1.0


def test_difficulty_changes():
    c = BayesianConfig()
    c.set_level(0)
    low = c.max_den
    c.set_level(6)
    high = c.max_den
    assert high > low


def test_answer_in_domain():
    from fractions import Fraction
    t = BayesianConditioning()
    for _ in range(20):
        ex = t.generate_example()
        f = Fraction(ex.answer)
        assert f.numerator > 0
        assert f.denominator > 1
        assert 0 < f < 1


def test_metadata_json():
    import json
    t = BayesianConditioning()
    ex = t.generate_example()
    json.dumps(ex.metadata.to_dict() if hasattr(ex.metadata, "to_dict") else dict(ex.metadata))


def test_deterministic_seed():
    random.seed(1234)
    t1 = BayesianConditioning()
    a1 = t1.generate_entry()
    random.seed(1234)
    t2 = BayesianConditioning()
    a2 = t2.generate_entry()
    assert a1.answer == a2.answer
