from contextlib import contextmanager

from nltk import CFG
import pytest

from reasoning_core.tasks import grammar as grammar_tasks
from reasoning_core.tasks.grammar import (
    ConstrainedContinuation,
    ConstrainedContinuationConfig,
    GrammarConfig,
    _exact_next_tokens_and_stop,
    exact_window_fills,
)


def test_constrained_continuation_config_scales_span_without_branch_explosion():
    config = ConstrainedContinuationConfig()
    config.set_level(6)

    assert config.min_k == 3
    assert config.max_k == 7
    assert config.n_types == 6
    assert config.n_terminals == 8
    assert config.max_num_rules == 14
    assert config.max_options == 85
    assert config.max_slot_checks == 32
    assert config.random_grammar_prob == 1
    assert config.free_form_grammar_prob == 0


def test_exact_window_fills_stops_at_state_limit():
    grammar = CFG.fromstring("S -> A A A\nA -> 'a' | 'b' | 'c' | 'd'")

    assert exact_window_fills(grammar, [], 3, max_states=16) == []
    assert len(exact_window_fills(grammar, [], 3, max_states=64)) == 64


def test_packed_recognizer_handles_recursive_ambiguity_without_tree_expansion():
    grammar = CFG.fromstring("S -> S S | 'a'")

    assert _exact_next_tokens_and_stop(grammar, []) == ({"a"}, False)
    assert _exact_next_tokens_and_stop(grammar, ["a", "a", "a"]) == ({"a"}, True)


def test_constrained_continuation_skips_oversized_sentences(monkeypatch):
    grammar = CFG.fromstring("S -> 'a' 'b' 'c'")
    outputs = iter(["a b c d e f", "a b c"])

    @contextmanager
    def fixed_grammar(*args, **kwargs):
        yield grammar

    class Generated:
        def __init__(self, text):
            self.text = text

        def __matmul__(self, key):
            return self.text

    monkeypatch.setattr(grammar_tasks, "resampled_grammar", fixed_grammar)
    monkeypatch.setattr(
        grammar_tasks,
        "gramforge_generate",
        lambda *args, **kwargs: Generated(next(outputs)),
    )

    config = GrammarConfig(
        min_k=3, max_k=3, max_tokens=3, min_options=1,
        bnf_operator_prob=1,
    )
    problem = ConstrainedContinuation(config).generate()

    assert problem.answer == "a b c"
    assert problem.metadata.n_candidates == 1
    assert " ::= " in problem.metadata.g
    assert len(problem.metadata.prefix) + problem.metadata.k + len(problem.metadata.suffix) == 3
    assert "<HOLE>" in problem.metadata.sentence


def test_constrained_continuation_gives_token_level_partial_credit():
    task = ConstrainedContinuation()
    entry = type("Entry", (), {"answer": "a b c", "__getitem__": lambda self, key: getattr(self, key)})()

    assert task.score_answer("a b c", entry) == 1.0
    assert task.score_answer("a x c", entry) == pytest.approx(2 / 3)
    assert task.score_answer("a b", entry) == pytest.approx(2 / 3)
    assert task.score_answer("", entry) == 0.0
