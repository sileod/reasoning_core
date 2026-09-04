import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

from reasoning_core.tasks.generated.wave9.temporal_logic_monitoring.temporal_logic_monitoring import (  # noqa: E501
    TemporalLogicMonitoring,
    evaluate_trace,
    eval_formula,
)


def _build():
    task = TemporalLogicMonitoring()
    task.config.set_level(0)
    return task


def test_gold_scores_one_every_level():
    task = TemporalLogicMonitoring()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0


def test_wrong_answers_do_not_score_one():
    task = _build()
    for _ in range(30):
        ex = task.generate_example()
        assert task.score_answer('', ex) == 0.0
        assert task.score_answer('garbage', ex) == 0.0
        assert task.score_answer('999', ex) == 0.0


def test_evaluator_matches_trace_truths():
    atom_true = {'a': [True, False, True, False], 'b': [False, True, True, False]}
    f_even = ('atom', 'a')
    assert eval_formula(f_even, atom_true, 0) is True
    assert eval_formula(f_even, atom_true, 1) is False
    f_X = ('X', ('atom', 'b'))
    assert eval_formula(f_X, atom_true, 0) is True
    assert eval_formula(f_X, atom_true, 2) is False  # position 3 false -> not next
    f_F = ('F', ('atom', 'a'))
    assert eval_formula(f_F, atom_true, 1) is True   # a holds at 2
    f_G = ('G', ('atom', 'a'))
    assert eval_formula(f_G, atom_true, 2) is False  # a fails at 3
    f_U = ('U', ('atom', 'a'), ('atom', 'b'))
    assert eval_formula(f_U, atom_true, 0) is True   # b at 1, a at 0
    assert eval_formula(f_U, atom_true, 3) is False


def test_answer_never_empty_or_full_across_levels():
    task = TemporalLogicMonitoring()
    seen = 0
    for level in range(7):
        task.config.set_level(level)
        for _ in range(15):
            ex = task.generate_example()
            seen += 1
            pos = sorted(ex.metadata.answer_positions)
            length = ex.metadata.length
            assert 0 < len(pos) < length
    assert seen >= 7 * 15


def test_answer_diversity():
    task = _build()
    answers = set()
    for _ in range(40):
        ex = task.generate_example()
        answers.add(ex.answer)
    assert len(answers) > 10


def test_metadata_json_serializable():
    import json
    task = _build()
    ex = task.generate_example()
    json.dumps(dict(ex.metadata))
