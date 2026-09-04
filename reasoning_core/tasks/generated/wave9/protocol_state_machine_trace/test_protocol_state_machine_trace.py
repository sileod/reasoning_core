import random

from reasoning_core.tasks.generated.wave9.protocol_state_machine_trace.protocol_state_machine_trace import (
    ProtocolStateMachineTrace,
    _simulate,
)


def _seed_and_run(level, seed=0):
    random.seed(seed)
    t = ProtocolStateMachineTrace()
    cfg = t.config_cls()
    cfg.set_level(level)
    t.config = cfg
    return t


def test_gold_scores_one_each_level():
    for level in range(7):
        random.seed(100 + level)
        t = ProtocolStateMachineTrace()
        cfg = t.config_cls()
        cfg.set_level(level)
        t.config = cfg
        for _ in range(20):
            x = t.generate_example()
            assert t.score_answer(x.answer, x) == 1.0


def test_answer_in_states():
    random.seed(42)
    t = ProtocolStateMachineTrace()
    x = t.generate_example()
    assert x.answer.startswith("s")
    assert x.answer in x.metadata["states"]


def test_junk_scores_zero():
    random.seed(1)
    t = ProtocolStateMachineTrace()
    x = t.generate_example()
    assert t.score_answer("", x) == 0.0
    assert t.score_answer("garbage", x) == 0.0
    assert t.score_answer(None, x) == 0.0


def test_difficulty_changes_config():
    t = ProtocolStateMachineTrace()
    c0 = t.config_cls()
    c0.set_level(0)
    c6 = t.config_cls()
    c6.set_level(6)
    assert c6.trace_len > c0.trace_len
    assert c6.n_states >= c0.n_states


def test_summary_presence():
    assert "protocol" in ProtocolStateMachineTrace.summary.lower()


def test_simulate_reproduces_gold_from_metadata():
    random.seed(99)
    task = ProtocolStateMachineTrace()
    seen = set()
    for _ in range(300):
        x = task.generate_example()
        m = x.metadata
        stars = {s: {} for s in m["states"]}
        for key, tgt in m["target_of"].items():
            s, e = key.split("|")
            stars[s][e] = tgt
        tos = {s: {} for s in m["states"]}
        for key, t in m["timeout_of"].items():
            s, e = key.split("|")
            tos[s][e] = t
        guards = {s: set(m["guard_of"][s]) for s in m["states"]}
        final = _simulate(m["states"], m["events"], stars, guards, tos,
                          m["start"], m["trace"])
        assert final == x.answer
        assert final in m["states"]
        seen.add(final)
        assert task.score_answer(x.answer, x) == 1.0
        assert task.score_answer("zz", x) == 0.0
    assert len(seen) > 1
