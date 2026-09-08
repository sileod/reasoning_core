import random

from reasoning_core.tasks.generated.wave10.scoped_instruction.scoped_instruction import (
    ScopedInstruction,
    ScopedInstructionConfig,
)


def test_gold_scores_one():
    random.seed(1)
    t = ScopedInstruction()
    for _ in range(50):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_junk_scores_less():
    random.seed(2)
    t = ScopedInstruction()
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("garbage", e) == 0.0


def test_answer_domain():
    random.seed(3)
    t = ScopedInstruction()
    for _ in range(100):
        e = t.generate_example()
        p = e.metadata.payload
        if p.mode == "section":
            vals = [int(x) for x in e.answer.split(",")]
            assert len(vals) == len(p.new)
            assert all(v > sum(p.base) for v in vals)
        else:
            assert int(e.answer) == p.target


def test_difficulty_scaling():
    c = ScopedInstructionConfig()
    c.set_level(0)
    d0 = c.section_len
    c.set_level(6)
    d6 = c.section_len
    assert d6 >= d0
