from contextlib import contextmanager

from nltk import CFG
import pytest

from reasoning_core.tasks import grammar as grammar_tasks
from reasoning_core.tasks.grammar import (
    ConstrainedContinuation,
    GrammarConfig,
    exact_window_fills,
)


def test_exact_window_fills_stops_at_state_limit():
    grammar = CFG.fromstring("S -> A A A\nA -> 'a' | 'b' | 'c' | 'd'")

    assert exact_window_fills(grammar, [], 3, max_states=16) == []
    assert len(exact_window_fills(grammar, [], 3, max_states=64)) == 64


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
        min_k=3, max_k=3, max_tokens=3, min_options=1, lean_style_prob=1,
    )
    problem = ConstrainedContinuation(config).generate()

    assert problem.answer == "a b c"
    assert problem.metadata.n_candidates == 1
    assert " := " in problem.metadata.g
    assert len(problem.metadata.prefix) + problem.metadata.k + len(problem.metadata.suffix) == 3
    assert "<HOLE>" in problem.metadata.sentence


def test_constrained_continuation_gives_token_level_partial_credit():
    task = ConstrainedContinuation()
    entry = type("Entry", (), {"answer": "a b c", "__getitem__": lambda self, key: getattr(self, key)})()

    assert task.score_answer("a b c", entry) == 1.0
    assert task.score_answer("a x c", entry) == pytest.approx(2 / 3)
    assert task.score_answer("a b", entry) == pytest.approx(2 / 3)
    assert task.score_answer("", entry) == 0.0
