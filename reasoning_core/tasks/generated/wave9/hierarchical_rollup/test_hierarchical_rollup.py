import random
import ast

from reasoning_core.tasks.generated.wave9.hierarchical_rollup.hierarchical_rollup import (
    HierarchicalRollup,
    HierarchicalRollupConfig,
    _build_tree,
    _rollup,
)


def _rec_check(v, excl, ask_node, vals, children):
    if v == excl:
        return 0
    if v == ask_node:
        chs = [c for c in children.get(v, []) if c != excl]
    else:
        chs = children.get(v, [])
    return vals.get(v, 0) + sum(_rec_check(c, excl, ask_node, vals, children) for c in chs)


def test_generate_and_score():
    random.seed(123)
    task = HierarchicalRollup()
    for _ in range(50):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0
        assert task.score_answer("", e) < 1.0
        assert task.score_answer("abc", e) < 1.0
        gold = ast.literal_eval(e.answer)
        assert isinstance(gold, int) and gold >= 0


def test_answer_correct_override():
    random.seed(7)
    task = HierarchicalRollup()
    n = 0
    while n < 20:
        task.config.set_level(0)
        e = task.generate_entry()
        if e.metadata.mode != "override":
            continue
        n += 1
        if e.metadata.overrides:
            (k, v) = next(iter(e.metadata.overrides.items()))
            parent, children = _build_tree(len(e.metadata.values), 3)
            children = {}
            # reconstruct children from metadata nodes
            for line in e.metadata.nodes:
                left, right = line.split(" -> ")
                children[left] = right.split(", ")
            vals = dict(e.metadata.values)
            expected = _rollup(e.metadata.ask_node, vals, dict(e.metadata.overrides), children)
            assert int(e.answer) == expected


def test_answer_correct_subtree():
    random.seed(11)
    task = HierarchicalRollup()
    n = 0
    while n < 20:
        task.config.set_level(0)
        e = task.generate_entry()
        if e.metadata.mode != "subtree":
            continue
        n += 1
        children = {}
        for line in e.metadata.nodes:
            left, right = line.split(" -> ")
            children[left] = right.split(", ")
        vals = dict(e.metadata.values)
        expected = _rec_check(e.metadata.ask_node, e.metadata.excl, e.metadata.ask_node, vals, children)
        assert int(e.answer) == expected


def test_difficulty():
    cfg = HierarchicalRollupConfig()
    cfg.set_level(0)
    base = int(cfg.n_nodes)
    cfg2 = HierarchicalRollupConfig()
    cfg2.set_level(6)
    assert int(cfg2.n_nodes) >= base


def test_scores_garbage_zero():
    random.seed(5)
    task = HierarchicalRollup()
    e = task.generate_entry()
    for bad in ["-", "1.5.5", "None", "True", "[]"]:
        assert task.score_answer(bad, e) == 0.0
