import random

import pytest

from reasoning_core.tasks.mutated.wave0.m08_elimination_depth.elimination_depth_chain import \
    EliminationDepthChain


def _ex(level):
    t = EliminationDepthChain()
    t.config.set_level(level)
    return t.generate_entry()


def test_gold_scores_one_all_levels():
    for level in (0, 1, 2, 3, 4, 5):
        for _ in range(5):
            t = EliminationDepthChain()
            t.config.set_level(level)
            x = t.generate_entry()
            assert t.score_answer(x.answer, x) == 1.0


def test_answer_distribution_broad():
    random.seed(3396779050)
    answers = {_ex(5).answer for _ in range(30)}
    assert len(answers) > 10


def test_diagnostic_matches_target():
    t = EliminationDepthChain()
    t.config.set_level(5)
    x = t.generate_entry()
    assert x.metadata["diagnostic_depth"] == x.metadata["target_depth"]
    assert x.metadata["target_depth"] >= 1


def test_dimension_approx_fixed():
    dims = set()
    for level in (0, 2, 5):
        t = EliminationDepthChain()
        t.config.set_level(level)
        dims.add(t.generate_entry().metadata["num_vars"])
    assert dims == {8}


def test_depth_monotonic():
    diags = []
    for level in (0, 2, 5):
        t = EliminationDepthChain()
        t.config.set_level(level)
        diags.append(t.generate_entry().metadata["diagnostic_depth"])
    assert diags[0] < diags[1] < diags[2]


def test_wrong_answer_does_not_score_one():
    t = EliminationDepthChain()
    t.config.set_level(3)
    x = t.generate_entry()
    true_val = float(x.answer)
    assert t.score_answer(str(true_val + 1), x) != 1.0
