import random

import pytest

from reasoning_core.tasks.mutated.wave0.m15_conjunctive_logic.conjunctive_logic import (
    ConjunctiveLogicQA,
)
from reasoning_core.tasks.logic_depth import (
    Atom,
    Theory,
    chase,
)


def _make_example(level):
    task = ConjunctiveLogicQA()
    task.config.set_level(level)
    entry = task.generate_entry()  # avoid tokenizer (tiktoken/regex pytest quirk)
    return task, entry


def test_roundtrip_all_levels():
    for level in (0, 2, 5):
        task, ex = _make_example(level)
        prompt = task.render_prompt(ex.metadata)
        assert ex.answer.strip()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_wrong_answer_scores_zero():
    task, ex = _make_example(0)
    assert task.score_answer("someone-not-in-domain", ex) == 0.0
    assert task.score_answer("  " + ex.answer.upper() + ". ", ex) == 1.0  # case/punct leniency


def test_branch_independence_and_nonlexical_conjuncts():
    random.seed(11)
    task = ConjunctiveLogicQA()
    case = task._build_case()
    assert case is not None
    res = chase(case.theory, max_depth=None)
    assert not res.inconsistent
    for atom in case.conjuncts:
        assert atom in res.closure
        assert res.derivations[atom].depth >= 1
        assert atom not in case.theory.facts
    # Removing branch-1 base facts must break h1.
    x = case.x
    b1_facts = {Atom(f.pred, (x,)) for f in case.theory.facts[:2]}
    sub1 = Theory(
        [f for f in case.theory.facts if f not in b1_facts],
        case.theory.rules, [], case.theory.pred_sigs, case.theory.entities,
        case.theory.domain_pack,
    )
    r1 = chase(sub1, max_depth=None)
    assert not r1.inconsistent
    assert case.h1 not in r1.closure
