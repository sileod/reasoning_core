import json

import pytest

from reasoning_core.tasks.generated.response_contracts import (
    ConditionalResponseContract,
    ProtectedSpanTransformation,
    SchemaBoundQuery,
)
from reasoning_core.tasks.generated.structured_extraction import EvidenceSufficiency, TypedRelationExtraction


TASKS = [
    TypedRelationExtraction,
    EvidenceSufficiency,
    SchemaBoundQuery,
    ConditionalResponseContract,
    ProtectedSpanTransformation,
]


@pytest.mark.parametrize("task_cls", TASKS)
def test_generated_ultramix_reference_and_wrong_answer(task_cls):
    task = task_cls()
    for _ in range(8):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1
        assert task.score_answer("__definitely_wrong__", entry) < 1


@pytest.mark.parametrize("task_cls", [TypedRelationExtraction, EvidenceSufficiency, SchemaBoundQuery])
def test_generated_ultramix_json_scoring_is_key_order_insensitive(task_cls):
    task = task_cls()
    entry = task.generate_example()
    pretty = json.dumps(json.loads(entry.answer), indent=2, sort_keys=True)
    assert task.score_answer(pretty, entry) == 1


def test_generated_ultramix_difficulty_changes_structure():
    for task_cls, field in [
        (TypedRelationExtraction, "n_sentences"),
        (EvidenceSufficiency, "n_evidence"),
        (SchemaBoundQuery, "n_rows"),
        (ConditionalResponseContract, "n_records"),
        (ProtectedSpanTransformation, "n_items"),
    ]:
        task = task_cls()
        base = getattr(task.config, field)
        task.config.set_level(4)
        assert getattr(task.config, field) > base
