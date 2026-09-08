from reasoning_core.tasks.generated.wave10.correction_aware_summary.correction_aware_summary import (
    CorrectionAwareSummary,
)


def test_generate_and_score():
    task = CorrectionAwareSummary()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0
            assert task.score_answer("", e) < 1.0
            assert task.score_answer("garbage", e) < 1.0


def test_metadata_json_serializable():
    import json

    task = CorrectionAwareSummary()
    e = task.generate_example()
    json.dumps(e.metadata)
