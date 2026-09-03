import random

from reasoning_core.tasks.generated.wave8.virtual_method_dispatch.virtual_method_dispatch import (
    VirtualMethodDispatch, VirtualMethodDispatchConfig, _resolve_for_test,
)


def _gold_answer(entry):
    hier = dict(entry.metadata["hierarchy"])
    overrides = [tuple(o) for o in entry.metadata["overrides"]]
    runtime = entry.metadata["runtime"]
    method = entry.metadata["method"]
    return _resolve_for_test(hier, overrides, runtime, method)


def test_gold_scores_and_resolution_consistent():
    for level in (0, 3, 6):
        task = VirtualMethodDispatch()
        for _ in range(50):
            task.config.set_level(level)
            entry = task.generate_example()
            assert entry.answer == entry.metadata["answer_class"]
            assert entry.answer == _gold_answer(entry)
            assert task.score_answer(entry.answer, entry) == 1.0


def test_wrong_answers_score_zero():
    task = VirtualMethodDispatch()
    for _ in range(30):
        entry = task.generate_example()
        wrong = entry.answer + "_x"
        assert task.score_answer(wrong, entry) == 0.0
        assert task.score_answer("", entry) == 0.0
        assert task.score_answer("9", entry) == 0.0


def test_answervaries_across_examples():
    task = VirtualMethodDispatch()
    answers = set()
    for _ in range(120):
        entry = task.generate_example()
        answers.add(entry.answer)
    assert len(answers) > 3


def test_summary_present():
    assert len(VirtualMethodDispatch.summary) > 10
