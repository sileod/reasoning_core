from itertools import product

from greenery import parse as gparse

import reasoning_core.tasks.regex as regex_module
from reasoning_core.tasks.regex import (
    RegexReasoning,
    RegexReasoningConfig,
    _distinct_equivalent_rendering,
    _reduced_union_superset,
)


def test_equivalent_rendering_is_distinct_and_semantically_equal():
    regex = "(a|b)c"
    fsm = gparse(regex).to_fsm()
    equivalent = _distinct_equivalent_rendering(regex, fsm)

    assert equivalent != regex
    assert fsm.equivalent(gparse(equivalent).to_fsm())


def test_reduced_superset_does_not_embed_source_union_arm():
    regex = "a|b"
    fsm = gparse(regex).to_fsm()
    superset = _reduced_union_superset(regex, fsm, "c")

    assert regex not in superset
    assert fsm.issubset(gparse(superset).to_fsm())
    assert not fsm.equivalent(gparse(superset).to_fsm())


def _assert_stored_answer_is_shortlex_minimum(monkeypatch, regex_a, regex_b, alphabet):
    fa, fb = gparse(regex_a).to_fsm(), gparse(regex_b).to_fsm()
    monkeypatch.setattr(regex_module, "_sample_pair", lambda *args: (regex_a, fa, regex_b, fb))
    monkeypatch.setattr(regex_module.random, "choice", lambda choices: "distinguishing")
    task = RegexReasoning(RegexReasoningConfig(n_alpha=len(alphabet)))

    entry = task.generate_entry()
    witnesses = [
        "".join(chars)
        for length in range(len(entry.answer) + 1)
        for chars in product(alphabet, repeat=length)
        if fa.accepts("".join(chars)) != fb.accepts("".join(chars))
    ]

    assert entry.answer == min(witnesses, key=lambda word: (len(word), word))


def test_distinguishing_answer_is_lexicographically_first_shortest(monkeypatch):
    _assert_stored_answer_is_shortlex_minimum(monkeypatch, "b|c", "a", "abc")


def test_distinguishing_answer_is_nonempty(monkeypatch):
    fa, fb = gparse("a*").to_fsm(), gparse("a+").to_fsm()
    monkeypatch.setattr(
        regex_module, "_sample_pair", lambda *args: ("a*", fa, "a+", fb)
    )
    monkeypatch.setattr(regex_module.random, "choice", lambda choices: "distinguishing")

    task = RegexReasoning(RegexReasoningConfig(n_alpha=2))

    assert task.generate_entry() is None
