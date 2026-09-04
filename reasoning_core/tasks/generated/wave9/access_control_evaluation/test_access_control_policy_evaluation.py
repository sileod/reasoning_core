import random

from reasoning_core.tasks.generated.wave9.access_control_policy_evaluation.access_control_policy_evaluation import (
    AccessControlEvaluation,
    _traverse,
    _decision,
)


def _fresh(level):
    t = AccessControlEvaluation()
    t.config.set_level(level)
    return t


def test_roundtrip_all_levels():
    random.seed(7)
    for level in (0, 1, 2, 3, 4, 5, 6):
        for _ in range(40):
            t = _fresh(level)
            e = t.generate_entry()
            assert t.score_answer(e.answer, e) == 1.0
            assert set(e.answer.split(",")) <= {"allow", "deny", "undefined"}


def test_answer_lengths_vary():
    random.seed(11)
    seen = set()
    for _ in range(200):
        t = _fresh(random.randint(0, 6))
        e = t.generate_entry()
        seen.add(e.answer)
    assert len(seen) > 30


def test_inheritance_matches_reference():
    random.seed(3)
    for _ in range(100):
        t = _fresh(random.randint(0, 6))
        e = t.generate_entry()
        m = e.metadata
        eff = [_decision(m.groups, m.user, p, m.memberships, m.precedence) for p in m.acts]
        assert ",".join(eff) == e.answer


def test_scorer_rejects_junk():
    t = AccessControlEvaluation()
    t.config.set_level(0)
    e = t.generate_entry()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("garbage", e) == 0.0
    assert t.score_answer("ALLOW,ALLOW,ALLOW", e) != 1.0 or e.answer == "allow,allow,allow"
