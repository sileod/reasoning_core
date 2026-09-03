import random
import ast

import pytest

from reasoning_core.tasks.generated.wave8.petri_enabled_transitions.petri_enabled_transitions import (
    PetriEnabledTransitions,
    _parse_enabled,
    _enabled,
)


def _place_answer(text):
    s = text.strip().lower()
    if s == "none":
        return ()
    return tuple(sorted(x.strip().strip('"\'') for x in s.split(",") if x.strip()))


def _gold_from_entry(entry):
    return _place_answer(entry.answer)


def test_gold_scores_one():
    random.seed(1701447588)
    task = PetriEnabledTransitions()
    for _ in range(40):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    random.seed(1701447588)
    task = PetriEnabledTransitions()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("not a valid answer at all", e) == 0.0
    assert task.score_answer(None, e) == 0.0


def test_wrong_subset_scores_zero():
    random.seed(1701447588)
    task = PetriEnabledTransitions()
    for _ in range(40):
        e = task.generate_example()
        gold = _gold_from_entry(e)
        if len(gold) >= 2:
            wrong = list(gold)
            wrong.pop()  # strict subset -> wrong answer
            assert task.score_answer(", ".join(wrong), e) == 0.0
            return
    pytest.fail("no multi-enabled example produced")


def test_enabled_verifier_agrees():
    random.seed(1701447588)
    for _ in range(200):
        n_places = random.randint(3, 8)
        n_trans = random.randint(3, 8)
        max_tokens = random.randint(3, 8)
        max_weight = random.randint(1, 3)
        places = [f"p{i}" for i in range(n_places)]
        trans = [f"t{i}" for i in range(n_trans)]
        tokens = {p: random.randint(0, max_tokens) for p in places}
        pre = {}
        for t in trans:
            k = random.randint(1, min(3, n_places))
            chosen = random.sample(places, k)
            pre[t] = tuple((p, random.randint(1, max_weight)) for p in chosen)
        enabled = set(_enabled(tokens, pre, trans))
        for t in trans:
            weight = sum(w for (p, w) in pre.get(t, ())) if pre.get(t) else 0
            manually = all(tokens.get(p, 0) >= w for (p, w) in pre.get(t, ()))
            assert (t in enabled) == manually
        # brute force sanity: recompute from scratch
        brute = set()
        for t in trans:
            if all(tokens.get(p, 0) >= w for (p, w) in pre.get(t, ())):
                brute.add(t)
        assert brute == enabled


def test_difficulty_changes():
    task = PetriEnabledTransitions()
    c0 = task.config_cls()
    c0.set_level(0)
    c6 = task.config_cls()
    c6.set_level(6)
    assert c0.n_trans < c6.n_trans


def test_parse_roundtrip():
    assert _parse_enabled("none") == ()
    assert _parse_enabled(" t1, t0, t2 ") == ("t0", "t1", "t2")
    assert _parse_enabled('"t1", "t3"') == ("t1", "t3")
    assert _parse_enabled("") is None
    assert _parse_enabled(None) is None
