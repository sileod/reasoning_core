import random

from reasoning_core.tasks.generated.wave11.causal_identification.causal_identification import (
    CausalIdentification,
    CausalIdentificationConfig,
)


def test_gold_scores_one():
    random.seed(7)
    t = CausalIdentification()
    for _ in range(40):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers_do_not_score_one():
    random.seed(11)
    t = CausalIdentification()
    scored = 0
    for _ in range(40):
        e = t.generate_example()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("garbage", e) == 0.0
        # a random different answer
        cands = e.metadata.payload["candidates"]
        import itertools
        wrong_answers = set()
        n = len(cands)
        for r in range(n + 1):
            for combo in itertools.combinations(cands, r):
                wrong_answers.add(",".join(sorted(combo)) if combo else "none")
        wrong_answers.discard(e.answer)
        for w in list(wrong_answers)[:3]:
            assert t.score_answer(w, e) == 0.0
        scored += 1
    assert scored > 0


def test_difficulty_changes_config():
    random.seed(3)
    cfg0 = CausalIdentificationConfig()
    cfg0.set_level(0)
    cfg6 = CausalIdentificationConfig()
    cfg6.set_level(6)
    assert cfg6.n_ops >= cfg0.n_ops
    assert cfg6.n_candidates >= cfg0.n_candidates


def test_answer_variety():
    random.seed(5)
    t = CausalIdentification()
    answers = set()
    for _ in range(60):
        e = t.generate_example()
        answers.add(e.answer)
    assert len(answers) > 1


def test_answer_widely_spread():
    random.seed(123)
    t = CausalIdentification()
    single = {}
    total = 200
    for _ in range(total):
        e = t.generate_example()
        assert e.answer != ""
        key = e.answer
        single[key] = single.get(key, 0) + 1
    assert len(single) > 1
    worst = max(single.values())
    assert worst / total < 0.8


def test_gold_possible_across_levels():
    random.seed(99)
    t = CausalIdentification()
    for level in (0, 3, 6):
        t.config.set_level(level)
        for _ in range(30):
            e = t.generate_example()
            assert e.answer == "none" or all(c.startswith("C") for c in e.answer.split(","))
