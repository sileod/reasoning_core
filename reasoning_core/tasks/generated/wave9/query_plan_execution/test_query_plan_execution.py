import reasoning_core.tasks.generated.wave9.query_plan_execution.query_plan_execution as m


def test_roundtrip_scores_one():
    task = m.QueryPlanExecution()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_zero():
    task = m.QueryPlanExecution()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer("", ex) == 0.0
        assert task.score_answer("garbage here", ex) == 0.0


def test_answer_matches_reexecution():
    task = m.QueryPlanExecution()
    for _ in range(50):
        ex = task.generate_example()
        assert m._execute(ex.metadata) == ex.answer


def test_modes_appear():
    task = m.QueryPlanExecution()
    modes = set()
    for _ in range(60):
        modes.add(task.generate_example().metadata.mode)
    assert modes == {"count", "sum", "distinct"}


def test_answers_vary():
    task = m.QueryPlanExecution()
    answers = {task.generate_example().answer for _ in range(60)}
    assert len(answers) > 15


def test_difficulty_changes_config():
    cfg = m.QueryPlanExecutionConfig()
    base = (cfg.n_r, cfg.n_s, cfg.n_dom, cfg.a_max)
    cfg.set_level(4)
    high = (cfg.n_r, cfg.n_s, cfg.n_dom, cfg.a_max)
    assert high != base
