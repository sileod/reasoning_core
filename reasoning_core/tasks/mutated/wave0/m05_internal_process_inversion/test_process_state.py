import random
import pytest

from reasoning_core.tasks.mutated.wave0.m05_internal_process_inversion.process_state import (
    ProcessState,
    ProcessStateConfig,
    _build_chain,
)


def _chain_simulate(base, steps):
    cur = base
    states = [cur]
    for op, k in steps:
        if op == "add":
            cur += k
        elif op == "sub":
            cur -= k
        elif op == "mul":
            cur *= k
        else:
            cur //= k
        states.append(cur)
    return states


def test_gold_scoring_all_levels():
    random.seed(2033032770)
    for level in (0, 1, 2, 3, 4, 5):
        t = ProcessState()
        t.config.set_level(level)
        for _ in range(50):
            ex = t.generate_example()
            assert t.score_answer(ex.answer, ex) == 1.0


def test_answer_matches_states():
    random.seed(11)
    t = ProcessState()
    for _ in range(200):
        ex = t.generate_example()
        m = ex.metadata
        states = _chain_simulate(m.base, m.steps)
        got = m.answer_val
        assert states[-1] == m.observed
        k = m.k
        if m.target == "before":
            assert got == states[k - 1]
        else:
            assert got == states[k]
        assert int(ex.answer) == got


def test_answer_is_internal():
    random.seed(22)
    t = ProcessState()
    counts = {}
    for _ in range(300):
        ex = t.generate_example()
        m = ex.metadata
        assert m.base != m.answer_val
        assert m.observed != m.answer_val
        counts[ex.answer] = counts.get(ex.answer, 0) + 1
    assert len(counts) > 30


def test_answers_vary_and_not_constant():
    random.seed(33)
    t = ProcessState()
    answers = set()
    for _ in range(300):
        answers.add(t.generate_example().answer)
    assert len(answers) > 30


def test_wrong_answers_score_low():
    random.seed(44)
    t = ProcessState()
    for _ in range(100):
        ex = t.generate_example()
        far = str(int(ex.answer) * 7 + 3)
        assert t.score_answer(far, ex) < 0.05
        assert t.score_answer("not a number", ex) == 0.0


def test_query_depth_scales():
    t = ProcessState()
    t.config.set_level(0)
    depths0 = [t.generate_example().metadata.forward_distance for _ in range(400)]
    random.seed(99)
    t2 = ProcessState()
    t2.config.set_level(5)
    depths5 = [t2.generate_example().metadata.forward_distance for _ in range(400)]
    assert min(depths5) >= min(depths0)
    assert max(depths5) > max(depths0)


def test_chain_builds_valid():
    random.seed(5)
    cfg = ProcessStateConfig()
    for _ in range(1000):
        steps, states, base = _build_chain(cfg)
        assert len(steps) == len(states) - 1
        assert all(isinstance(_, int) and _ > 0 for _ in states)
