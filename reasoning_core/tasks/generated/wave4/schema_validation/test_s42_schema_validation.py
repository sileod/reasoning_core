import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from reasoning_core.tasks.generated.wave4.s42_schema_validation.s42_schema_validation import (
    SchemaValidation, S42SchemaValidationConfig
)


def test_generate_and_score():
    t = SchemaValidation()
    for _ in range(200):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_difficulty_changes():
    c = S42SchemaValidationConfig()
    c.set_level(3)
    assert c.depth > S42SchemaValidationConfig().depth or c.n_keys > S42SchemaValidationConfig().n_keys


def test_invalid_answers():
    t = SchemaValidation()
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("garbage", e) == 0.0


def test_answer_is_valid_or_path():
    t = SchemaValidation()
    for _ in range(100):
        e = t.generate_example()
        if e.answer == "valid":
            assert t.score_answer("valid", e) == 1.0
        else:
            # answer is a dotted path (possibly a bare key) of the schema
            assert e.answer.replace(".", "").isalnum() or e.answer.replace(".", "").isdigit()
            assert not e.answer.endswith(".") and not e.answer.startswith(".")
            assert t.score_answer("valid", e) == 0.0
            assert t.score_answer("", e) == 0.0
