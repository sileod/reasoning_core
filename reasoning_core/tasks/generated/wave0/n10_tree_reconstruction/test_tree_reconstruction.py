import random

from reasoning_core.tasks.generated.wave0.n10_tree_reconstruction.tree_reconstruction import (
    TreeReconstruction,
    TreeReconstructionConfig,
)


def _gen(n, left, right):
    pass


def test_gold_scores_one():
    t = TreeReconstruction()
    for level in (0, 1, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers_score_zero():
    t = TreeReconstruction()
    t.config.set_level(0)
    for _ in range(20):
        e = t.generate_example()
        wrong = str((int(e.answer) + 1) % 1000)
        assert t.score_answer(wrong, e) == 0.0


def test_garbage_does_not_crash():
    t = TreeReconstruction()
    e = t.generate_example()
    assert t.score_answer("abc", e) == 0.0
    assert t.score_answer("", e) == 0.0
    assert t.score_answer(None, e) == 0.0


def test_answer_is_single_integer():
    t = TreeReconstruction()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_example()
            assert str(int(e.answer)) == e.answer


def test_root_is_first_preorder_element():
    t = TreeReconstruction()
    t.config.set_level(0)
    e = t.generate_example()
    payload = e.metadata.payload
    if e.metadata.qtype == "preinorder":
        assert payload["preorder"][0] == 0


def test_traversal_consistency():
    t = TreeReconstruction()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_example()
            payload = e.metadata.payload
            assert len(set(payload["inorder"])) == len(payload["inorder"])
            if e.metadata.qtype == "inpostorder":
                assert sorted(payload["inorder"]) == sorted(payload["postorder"])


def test_metadata_json_serializable():
    import json
    t = TreeReconstruction()
    t.config.set_level(2)
    for _ in range(10):
        e = t.generate_example()
        json.dumps(dict(e.metadata))


def test_config_difficulty_changes():
    c = TreeReconstructionConfig()
    c.set_level(0)
    base = c.n_nodes
    c.set_level(5)
    assert c.n_nodes > base


def test_difficulty_monotonic():
    prev = None
    for level in range(6):
        c = TreeReconstructionConfig()
        c.set_level(level)
        n = c.n_nodes
        if prev is not None:
            assert n >= prev
        prev = n
