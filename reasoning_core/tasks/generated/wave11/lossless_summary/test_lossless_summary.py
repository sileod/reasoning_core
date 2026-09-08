import random

from reasoning_core.tasks.generated.wave11.lossless_summary.lossless_summary import (
    LosslessSummary,
    LosslessSummaryConfig,
)


def test_roundtrip_default():
    random.seed(12345)
    task = LosslessSummary()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0


def test_roundtrip_all_levels():
    for level in range(7):
        task = LosslessSummary()
        task.config.set_level(level)
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_zero():
    task = LosslessSummary()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("garbage", ex) == 0.0
    assert task.score_answer("999, 999", ex) == 0.0


def test_answer_is_ordered_query_values():
    random.seed(99)
    task = LosslessSummary()
    for _ in range(20):
        ex = task.generate_example()
        m = ex.metadata
        q = [x.strip() for x in m.query_order.split(",")]
        body = m.narrative.split("A mixed relay round just ended. ")[1]
        clauses = body.split(", and ")
        score_map = {}
        for clause in clauses:
            name, _, rest = clause.partition(" scored ")
            score_map[name] = int(rest.split()[0])
        expected = [str(score_map[name]) for name in q]
        assert ex.answer == ", ".join(expected)


def test_gold_reproduces_target():
    random.seed(7)
    task = LosslessSummary()
    for _ in range(30):
        ex = task.generate_example()
        vals = [int(v) for v in ex.answer.split(",")]
        m = ex.metadata
        narrative = m.narrative
        q = [x.strip() for x in m.query_order.split(",")]
        for name, val in zip(q, vals):
            assert f"{name} scored {val} points" in narrative
            assert f"room" in narrative and "kicked off" in narrative


def test_prompt_readable():
    random.seed(3)
    task = LosslessSummary()
    ex = task.generate_example()
    p = task.render_prompt(ex.metadata)
    assert "separated by commas and nothing else" in p
    assert "Answer:" in p


def test_config_scales():
    base = LosslessSummaryConfig()
    base.set_level(0)
    hi = LosslessSummaryConfig()
    hi.set_level(6)
    assert hi.n_queried >= base.n_queried
    assert hi.n_distract >= base.n_distract


def test_answer_domain_positive_ints():
    task = LosslessSummary()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(10):
            ex = task.generate_example()
            vals = ex.answer.split(", ")
            assert len(vals) == len([x for x in ex.metadata.query_order.split(", ")])
            for v in vals:
                assert int(v) >= 0
            assert all(isinstance(int(v), int) for v in vals)


def test_varied_answers_across_examples():
    random.seed(42)
    task = LosslessSummary()
    answers = set()
    for _ in range(40):
        ex = task.generate_example()
        answers.add(ex.answer)
    assert len(answers) >= 30

