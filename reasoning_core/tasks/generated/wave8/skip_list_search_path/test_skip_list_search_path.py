import ast
import json
import random

from reasoning_core.tasks.generated.wave8.skip_list_search_path.skip_list_search_path import (
    SkipListSearchPath,
    SkipListSearchPathConfig,
    _search_visited,
)


def test_gold_scores_one():
    t = SkipListSearchPath()
    for level in (0, 1, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers_score_zero():
    t = SkipListSearchPath()
    t.config.set_level(0)
    for _ in range(30):
        e = t.generate_example()
        wrong = "[999999]"
        if e.answer == wrong:
            wrong = "[-1]"
        assert t.score_answer(wrong, e) == 0.0


def test_visited_matches_recomputation():
    t = SkipListSearchPath()
    t.config.set_level(2)
    for _ in range(30):
        e = t.generate_example()
        keys = [k for k, _ in e.metadata.nodes]
        heights = [h for _, h in e.metadata.nodes]
        expected = _search_visited(keys, heights, e.metadata.max_level, e.metadata.target, len(keys))
        assert e.metadata.visited == expected
        assert ast.literal_eval(e.answer) == expected


def test_visited_keys_are_less_than_target():
    t = SkipListSearchPath()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            for v in e.metadata.visited:
                assert v < e.metadata.target


def test_garbage_does_not_crash():
    t = SkipListSearchPath()
    e = t.generate_example()
    for bad in ("abc", "", None, "[]", "[x]", "{1}", "1,2"):
        assert t.score_answer(bad, e) == 0.0


def test_config_difficulty_changes():
    c = SkipListSearchPathConfig()
    c.set_level(0)
    base = c.n_nodes
    bm = c.max_level
    c.set_level(5)
    assert c.n_nodes > base
    assert c.max_level >= bm


def test_answer_value_range():
    t = SkipListSearchPath()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            keys = [k for k, _ in e.metadata.nodes]
            assert keys == sorted(keys)
            assert len(set(keys)) == len(keys)
            assert e.metadata.max_level >= 1


def test_metadata_json_serializable():
    t = SkipListSearchPath()
    t.config.set_level(2)
    for _ in range(10):
        e = t.generate_example()
        json.dumps(dict(e.metadata))


def test_answers_vary_across_examples():
    t = SkipListSearchPath()
    t.config.set_level(3)
    seen = set()
    for _ in range(200):
        e = t.generate_example()
        seen.add(tuple(e.metadata.visited))
    assert len(seen) >= 50
