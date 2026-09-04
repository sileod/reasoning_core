import importlib.util
from pathlib import Path

import pytest

_MODULE = Path(__file__).with_name("reaching_definitions.py")
_spec = importlib.util.spec_from_file_location("wave9_rd_test", _MODULE)
_MOD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_MOD)

ReachingDefs = _MOD.ReachingDefs
ReachingDefinitionsConfig = _MOD.ReachingDefinitionsConfig
_build_cfg = _MOD._build_cfg
_CFG = _MOD._CFG
_reachable = _MOD._reachable


def _make_example(level):
    task = ReachingDefs()
    task.config = ReachingDefinitionsConfig()
    task.config.apply_difficulty(level)
    return task.generate_entry()


def test_gold_scores_one():
    for level in (0, 1, 2, 3, 5, 6):
        for _ in range(5):
            e = _make_example(level)
            task = ReachingDefs()
            assert task.score_answer(e.answer, e) == 1.0


def test_empty_and_junk_do_not_score_one():
    task = ReachingDefs()
    for level in (0, 3, 6):
        for _ in range(5):
            e = _make_example(level)
            assert task.score_answer("", e) != 1.0
            if e.answer == "-":
                continue
            assert task.score_answer("junk", e) != 1.0


def test_difficulty_changes_config():
    c0 = ReachingDefinitionsConfig(); c0.apply_difficulty(0)
    c6 = ReachingDefinitionsConfig(); c6.apply_difficulty(6)
    assert c6.n_stmts > c0.n_stmts


def test_reachable():
    for _ in range(50):
        cfg = ReachingDefinitionsConfig()
        cfg.apply_difficulty(5)
        edges = _build_cfg(cfg.n_stmts, cfg.n_branches, cfg.n_loops)
        assert len(_reachable(edges, cfg.n_stmts)) == cfg.n_stmts


def test_loop_fixpoint_gold_reaches_query():
    task = ReachingDefs()
    task.config = ReachingDefinitionsConfig()
    task.config.apply_difficulty(5)
    for _ in range(20):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0


def test_answers_vary():
    task = ReachingDefs()
    task.config.set_level(3)
    answers = {task.generate_example().answer for _ in range(30)}
    assert len(answers) >= 5
