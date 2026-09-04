from reasoning_core.tasks.generated.wave1.s10_heap_simulation.heap_simulation import HeapSimulation, HeapSimulationConfig


def test_round_trip():
    task = HeapSimulation()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_gold_scores_one_all_levels():
    task = HeapSimulation()
    for level in range(7):
        cfg = HeapSimulationConfig()
        cfg.set_level(level)
        task.config = cfg
        for _ in range(20):
            x = task.generate_example()
            assert task.score_answer(x.answer, x) == 1.0


def test_garbage_scores_zero():
    task = HeapSimulation()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("junk", x) == 0.0
    assert task.score_answer("1,2,x", x) == 0.0


def test_wrong_answer_zero():
    task = HeapSimulation()
    x = task.generate_example()
    if not x.metadata.ask_index:
        forward = sorted([int(v) for v in x.answer.split(',')])
        backward = sorted(forward, reverse=True)
        if ','.join(str(v) for v in backward) != x.answer:
            assert task.score_answer(','.join(str(v) for v in backward), x) == 0.0


def test_difficulty_changes():
    cfg0 = HeapSimulationConfig()
    cfg5 = HeapSimulationConfig()
    cfg0.set_level(0)
    cfg5.set_level(5)
    assert cfg5.n_ops > cfg0.n_ops


def test_answer_varies():
    task = HeapSimulation()
    answers = set()
    for _ in range(60):
        x = task.generate_example()
        answers.add(x.answer)
    assert len(answers) >= 20


def test_ask_index_format():
    task = HeapSimulation()
    cfg = HeapSimulationConfig()
    cfg.set_level(3)
    task.config = cfg
    x = task.generate_example()
    assert x.metadata.ask_index
    assert ',' not in x.answer
