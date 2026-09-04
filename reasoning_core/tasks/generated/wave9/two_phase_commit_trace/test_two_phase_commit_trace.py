import random
import json

from reasoning_core.template import Task
from reasoning_core.tasks.generated.wave9.two_phase_commit_trace.two_phase_commit_trace import (
    TwoPhaseCommitTrace,
    TwoPhaseCommitConfig,
    _protocol,
    _verify,
)


def test_gold_scores_one():
    task = TwoPhaseCommitTrace()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_zero():
    task = TwoPhaseCommitTrace()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("XYZ", ex) == 0.0
    assert task.score_answer(12, ex) == 0.0


def test_answer_matches_protocol():
    task = TwoPhaseCommitTrace()
    for _ in range(40):
        ex = task.generate_example()
        m = ex.metadata
        votes_int = {int(k): (None if v == "none" else v) for k, v in m.votes.items()}
        expected = _protocol(votes_int, True, set(m.failures))
        ans = "".join("C" if expected[p] == "commit" else "A" for p in m.participants)
        assert ex.answer == ans
        assert all(expected[p] == m.final[str(p)] for p in m.participants)


def test_answer_length():
    task = TwoPhaseCommitTrace()
    for _ in range(20):
        ex = task.generate_example()
        assert len(ex.answer) == ex.metadata.n_participants
        assert set(ex.answer) <= {"C", "A"}


def test_difficulty_changes_config():
    base = TwoPhaseCommitConfig()
    high = TwoPhaseCommitConfig()
    base.set_level(0)
    high.set_level(6)
    assert high.n_participants > base.n_participants
    assert high.max_failures >= base.max_failures


def test_distractors_differ():
    task = TwoPhaseCommitTrace()
    for _ in range(20):
        ex = task.generate_example()
        dists = list(task.distractor_candidates(ex))
        assert dists
        assert ex.answer not in dists


def test_metadata_json_serializable():
    task = TwoPhaseCommitTrace()
    for _ in range(10):
        ex = task.generate_example()
        json.dumps(dict(ex.metadata))


def test_reproducible_under_seed():
    random.seed(2867049114)
    a = TwoPhaseCommitTrace().generate_example()
    random.seed(2867049114)
    b = TwoPhaseCommitTrace().generate_example()
    assert a.answer == b.answer
    assert a.metadata.payload == b.metadata.payload


def test_prompt_determines_answer():
    task = TwoPhaseCommitTrace()
    for _ in range(10):
        ex = task.generate_example()
        prompt = task.render_prompt(ex.metadata)
        assert "C" in prompt
        assert "A" in prompt
