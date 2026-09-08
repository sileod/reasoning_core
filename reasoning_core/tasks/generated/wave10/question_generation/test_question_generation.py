import random

from reasoning_core.tasks.generated.wave10.question_generation.question_generation import (
    QuestionGeneration, QuestionGenerationV1Config, _normalize,
)


def _normalize_answer(ans):
    return _normalize(ans)


def test_roundtrip_score():
    task = QuestionGeneration()
    random.seed(1)
    counts = {}
    for _ in range(60):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0
        assert task.score_answer("", ex) == 0.0
        assert task.score_answer("junk", ex) == 0.0
        counts[ex.answer] = counts.get(ex.answer, 0) + 1
    assert len(counts) > 10


def test_all_roles_are_bracketed():
    task = QuestionGeneration()
    random.seed(2)
    seen = set()
    for _ in range(300):
        ex = task.generate_example()
        sentence = ex.metadata.sentence
        assert "[" in sentence and "]" in sentence
        seen.add(ex.metadata.constituent_role)
    assert seen == {"subj", "obj", "time", "place", "reason", "manner"}


def test_difficulty_changes():
    c = QuestionGenerationV1Config()
    c.apply_difficulty(0)
    lo = c.n_context
    c.apply_difficulty(6)
    hi = c.n_context
    assert hi > lo


def test_metadata_json_serializable():
    import json
    task = QuestionGeneration()
    random.seed(3)
    ex = task.generate_example()
    json.dumps(dict(ex.metadata))
