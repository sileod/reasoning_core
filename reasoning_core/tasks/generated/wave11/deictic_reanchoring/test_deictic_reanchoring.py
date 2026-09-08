"""Focused tests for the DeicticReanchoring task."""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from reasoning_core.template import Entry, edict
from deictic_reanchoring import (
    DeicticReanchoring,
    _normalize,
    _reanchor_clauses,
    _verify_answer,
)


def test_gold_scores_one_every_level():
    task = DeicticReanchoring()
    for level in (0, 1, 2, 3, 4, 5, 6):
        task.config.set_level(level)
        for _ in range(5):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0, (level, e.answer)


def test_garbage_not_scored_one():
    task = DeicticReanchoring()
    e = task.generate_example()
    assert task.score_answer("", e) < 1.0
    assert task.score_answer("nonsense output", e) < 1.0


def test_difficulty_changes_config():
    task = DeicticReanchoring()
    task.config.set_level(0)
    lo = task.config.n_people
    task.config.set_level(5)
    hi = task.config.n_people
    assert hi >= lo


def test_reproducible_under_fixed_seed():
    random.seed(1234)
    e1 = DeicticReanchoring().generate_example()
    random.seed(1234)
    e2 = DeicticReanchoring().generate_example()
    assert e1.answer == e2.answer
    assert e1.metadata["source_utterance"] == e2.metadata["source_utterance"]


def test_answer_varies_across_examples():
    task = DeicticReanchoring()
    answers = {task.generate_example().answer for _ in range(30)}
    assert len(answers) > 5


def test_roundtrip_verify():
    # Direct construction sanity: clauses match schedule referents.
    sched = [[["x"], ["y"], ["z"]], [["p"], ["q"], ["r"]]]
    clauses = _reanchor_clauses(1, "Alice", "Bob", sched, 0, 1, 0)
    assert clauses == [
        "Alice is at ['x'] on Day 1", "Bob is at ['p'] on Day 1"]
    assert _verify_answer(1, "Alice", "Bob", sched, 0, 1, 1, clauses)
    assert not _verify_answer(1, "Alice", "Bob", sched, 0, 1, 1,
                              ["Alice is at y on Day 1",
                               "Bob is at p on Day 1"])


def test_metadata_json_serializable():
    import json
    task = DeicticReanchoring()
    e = task.generate_example()
    json.dumps({k: v for k, v in e.metadata.items()})
    json.dumps(e.metadata["payload"])


def test_prompt_mentions_answer_format():
    task = DeicticReanchoring()
    e = task.generate_example()
    prompt = task.render_prompt(e.metadata)
    assert "PLACE on Day N" in prompt
    assert e.metadata["source_utterance"] in prompt
