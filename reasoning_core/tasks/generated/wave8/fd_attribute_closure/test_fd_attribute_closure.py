import random

from reasoning_core.tasks.generated.wave8.fd_attribute_closure.fd_attribute_closure import (
    AttributeClosure,
    _closure,
)


def _parse_fds(lines, names):
    fds = []
    for ln in lines:
        lhs_s, rhs_s = ln.split(" -> ")
        lhs = frozenset(names.index(ch) for ch in lhs_s)
        rhs = frozenset(names.index(ch) for ch in rhs_s)
        fds.append((frozenset(lhs), frozenset(rhs)))
    return fds


def _parse_start(s, names):
    inner = s.strip("{}").replace(" ", "").replace(",", "")
    if not inner:
        return frozenset()
    return frozenset(names.index(ch) for ch in inner)


def test_gold_scores_one():
    t = AttributeClosure()
    t.config.set_level(2)
    for _ in range(50):
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0


def test_junk_scores_zero():
    t = AttributeClosure()
    t.config.set_level(2)
    x = t.generate_example()
    assert t.score_answer("", x) == 0.0
    assert t.score_answer("hello world", x) == 0.0
    assert t.score_answer("!!!", x) == 0.0


def test_closure_matches_reference():
    t = AttributeClosure()
    t.config.set_level(3)
    for _ in range(50):
        x = t.generate_example()
        names = [chr(ord("A") + int(x.metadata.n_atts) + i - int(x.metadata.n_atts)) for i in range(int(x.metadata.n_atts))]
        p = x.metadata.payload
        fds = _parse_fds(p["dependencies"].split("\n"), names)
        start = _parse_start(p["start"], names)
        clos = _closure(fds, start)
        gold = "".join(sorted(names[i] for i in clos))
        assert gold == x.answer


def test_raw_nontrivial_instances():
    t = AttributeClosure()
    t.config.set_level(6)
    seen = set()
    for _ in range(40):
        x = t.generate_example()
        assert x.answer not in seen or len(seen) < 5
        seen.add(x.answer)


def test_all_levels_generate():
    for lev in range(7):
        t = AttributeClosure()
        t.config.set_level(lev)
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0
