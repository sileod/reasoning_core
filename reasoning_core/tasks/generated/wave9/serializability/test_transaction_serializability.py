import random

from reasoning_core.tasks.generated.wave9.transaction_serializability.transaction_serializability import (
    Serializability, _serialize, _cycle_core, _precedence_edges,
)


def _task(level):
    task = Serializability()
    cfg = task.config_cls()
    cfg.set_level(level)
    task.config = cfg
    return task


def test_gold_scores_one_across_levels():
    random.seed(1234)
    for level in range(0, 7):
        task = _task(level)
        for _ in range(100):
            entry = task.generate_entry()
            assert task.score_answer(entry.answer, entry) == 1.0


def test_semantic_accepts_any_valid_serial():
    random.seed(99)
    task = _task(2)
    for _ in range(300):
        entry = task.generate_entry()
        if entry.answer.startswith("serial"):
            history = _parse(entry)
            ntr = entry.metadata["ntr"]
            order = _serialize(history, ntr)
            assert order is not None
            # a different valid topo must also score 1.0
            assert task.score_answer("serial " + ",".join("T{}".format(i) for i in order), entry) == 1.0
    random.seed(99)


def test_bad_answers_score_zero():
    random.seed(7)
    task = _task(2)
    for _ in range(200):
        entry = task.generate_entry()
        assert task.score_answer("", entry) < 1.0
        assert task.score_answer("garbage", entry) < 1.0
        if entry.answer.startswith("serial"):
            assert task.score_answer("nonserial T0,T1,T2", entry) < 1.0
            ntr = entry.metadata["ntr"]
            assert task.score_answer("serial " + ",".join("T{}".format(i) for i in ("x" for _ in range(ntr))), entry) < 1.0
            assert task.score_answer("serial T0", entry) < 1.0
        else:
            assert task.score_answer("serial T0,T1,T2", entry) < 1.0


def test_answers_vary():
    random.seed(5)
    task = _task(3)
    seen = set()
    serial_ct = 0
    nonserial_ct = 0
    for _ in range(200):
        entry = task.generate_entry()
        seen.add(entry.answer)
        if entry.answer.startswith("serial"):
            serial_ct += 1
        else:
            nonserial_ct += 1
    assert len(seen) > 90
    assert 0.2 < serial_ct / 200 < 0.8
    assert 0.2 < nonserial_ct / 200 < 0.8


def _parse(entry):
    history = []
    for h in entry.metadata["history"]:
        t = int(h[1]); op = h[2]; item = int(h[3:])
        history.append((t, op, item))
    return history
