from reasoning_core.tasks.generated.wave4.s44_spreadsheet_evaluation.spreadsheet_evaluation import (
    SpreadsheetEvaluation, SpreadsheetEvaluationConfig)


def test_generate_roundtrip():
    t = SpreadsheetEvaluation()
    e = t.generate_example()
    assert t.score_answer(e.answer, e) == 1.0
    assert t.score_answer("", e) < 1.0
    assert t.score_answer("zzz", e) < 1.0


def test_levels_work():
    for lvl in range(7):
        cfg = SpreadsheetEvaluationConfig()
        cfg.set_level(lvl)
        t = SpreadsheetEvaluation(config=cfg)
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0
        assert e.answer is not None


def test_prompt_has_answer_format():
    t = SpreadsheetEvaluation()
    e = t.generate_example()
    p = t.render_prompt(e.metadata)
    assert "=" in p


def test_cyclic_answer_alphabetical():
    import random
    found_cyclic = False
    for _ in range(50):
        cfg = SpreadsheetEvaluationConfig(cyclic_prob=1.0)
        cfg.set_level(2)
        t = SpreadsheetEvaluation(config=cfg)
        e = t.generate_example()
        if e.metadata.cyclic:
            found_cyclic = True
            names = e.answer.split(",")
            assert names == sorted(names)
    assert found_cyclic
