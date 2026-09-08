from collections import Counter

from easydict import EasyDict as edict
from nltk import CFG, ChartParser
import pytest

from reasoning_core.tasks import grammar as grammar_tasks
if not hasattr(grammar_tasks, "CFGSpan"):
    pytest.skip("CFGSpan is no longer an active task", allow_module_level=True)

from reasoning_core.tasks.grammar import (
    CFGSpan,
    GrammarConfig,
    ParsingDerivation,
    constituent_spans,
    mark_token,
    smallest_span_candidates,
)


def _tree():
    grammar = CFG.fromstring(
        """
        S -> A B
        A -> X Y
        X -> 'x'
        Y -> 'y'
        B -> Z W V
        Z -> 'z'
        W -> 'w'
        V -> 'v'
        """
    )
    return grammar, next(ChartParser(grammar).parse("x y z w v".split()))


def test_constituent_spans_and_smallest_candidates():
    _, tree = _tree()
    spans, end = constituent_spans(tree)
    assert end == 5
    assert {(x.label, x.start, x.end) for x in spans} >= {
        ("A", 0, 2), ("B", 2, 5), ("S", 0, 5),
    }

    candidates = smallest_span_candidates(tree)
    assert [(c.start, c.end) for c in candidates] == [
        (0, 2), (0, 2), (2, 5), (2, 5), (2, 5),
    ]
    assert mark_token(tree.leaves(), 3) == "x y z <M> w </M> v"


def test_cfg_span_generation_prompt_and_score(monkeypatch):
    grammar, tree = _tree()

    monkeypatch.setattr(
        grammar_tasks,
        "generate_parse",
        lambda _: edict(
            label="unambiguous",
            tokens=tree.leaves(),
            g="\n".join(map(str, grammar.productions())),
            parses=[tree],
            cot="unused",
        ),
    )
    task = CFGSpan(GrammarConfig())
    problem = task.generate_example()

    assert problem.answer == "2 5"
    assert problem.metadata.span_len == 3
    assert problem.metadata.marked_index in {2, 3}
    assert "<M>" in problem.metadata.marked_string
    assert "0-based, end-exclusive" in problem.prompt
    assert task.score_answer("start=2 end=5", problem) == 1.0
    assert task.score_answer("2 4", problem) == 0.0


def test_parsing_derivation_generation_prompt_and_score(monkeypatch):
    grammar, tree = _tree()
    examples = iter([
        edict(
            label="ambiguous",
            tokens=tree.leaves(),
            g="\n".join(map(str, grammar.productions())),
            parses=[tree, tree],
            cot="unused",
        ),
        edict(
            label="unambiguous",
            tokens=tree.leaves(),
            g="\n".join(map(str, grammar.productions())),
            parses=[tree],
            cot="unused",
        ),
    ])

    monkeypatch.setattr(
        grammar_tasks,
        "generate_parse",
        lambda _: next(examples),
    )
    problem = ParsingDerivation(GrammarConfig()).generate_example()

    assert problem.answer == "R0 R1 R2 R3 R4 R5 R6 R7"
    assert "R0: S -> A B" in problem.prompt
    assert "leftmost derivation" in problem.prompt
    assert problem.metadata.get("parses") is None
    assert problem.metadata.get("cot") is None
    assert ParsingDerivation().score_answer("Rules: R0, R1 R2 R3 R4 R5 R6 R7", problem) == 1.0
    assert 0.0 < ParsingDerivation().score_answer("R0 R2", problem) < 1.0
    assert ParsingDerivation().score_answer("R8 R9", problem) == 0.0


def test_cfg_span_filters_root_and_downsamples_length_two(monkeypatch):
    grammar = CFG.fromstring("S -> A B\nA -> 'a'\nB -> 'b'")
    tree = next(ChartParser(grammar).parse(["a", "b"]))
    task = CFGSpan(GrammarConfig())
    assert task._eligible(smallest_span_candidates(tree)) == []

    candidates = [edict(bucket="2", span_len=2, is_root=False, is_edge=False)]
    task.easy_accept = 0.0
    assert task._eligible(candidates) == []


def test_cfg_span_prefers_underrepresented_bucket():
    task = CFGSpan(GrammarConfig())
    task.span_hist = Counter({"3": 5, "4": 0})
    candidates = [
        edict(bucket="3", span_len=3),
        edict(bucket="4", span_len=4),
    ]
    assert task._pick(candidates).bucket == "4"
