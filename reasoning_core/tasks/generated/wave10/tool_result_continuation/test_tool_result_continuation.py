import random

from reasoning_core.tasks.generated.wave10.tool_result_continuation.tool_result_continuation import (
    ToolResultContinuation,
    ToolResultContinuationV2Config,
    _parse_answer,
    score_answer,
)

TASK = ToolResultContinuation()


def test_gold_scores_1():
    for _ in range(200):
        entry = TASK.generate_example()
        assert score_answer(None, entry.answer, entry) == 1.0, entry.answer


def test_state_matches_total_consistent():
    for _ in range(200):
        entry = TASK.generate_example()
        state, total = _parse_answer(entry.answer)
        assert isinstance(total, int) and total >= 0
        assert state in ("success", "empty", "partial", "failure")


def test_junk_scores_0():
    entry = TASK.generate_example()
    assert score_answer(None, entry.answer, entry) == 1.0
    for junk in ("", "bogus", "success", ":3", "partial:", "failure:-1", "7:success"):
        assert score_answer(None, junk, entry) == 0.0


def test_difficulty_increases_prompt():
    c0 = ToolResultContinuationV2Config()
    c0.set_level(0)
    c5 = ToolResultContinuationV2Config()
    c5.set_level(5)
    assert c5.n_calls_min >= c0.n_calls_min
    assert c5.n_calls_max >= c0.n_calls_max
    assert c5.target_max >= c0.target_max


def test_all_states_occur(seed=11):
    random.seed(seed)
    states = set()
    seen_totals = set()
    for _ in range(400):
        entry = TASK.generate_entry()
        state, total = _parse_answer(entry.answer)
        states.add(state)
        seen_totals.add(total)
    assert states == {"success", "empty", "partial", "failure"}
    assert len(seen_totals) > 5


def test_metadata_json_serializable():
    import json

    entry = TASK.generate_example()
    json.dumps(dict(entry.metadata))
    json.dumps(dict(entry.metadata.payload))
