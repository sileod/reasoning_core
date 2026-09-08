import json

from reasoning_core.tasks.regex import RegexInduction, RegexRetrieval, synthesize_shortest_regex


def test_synth_literal():
    assert synthesize_shortest_regex(["a"], ["b", "c"], "abc") == "a"


def test_synth_alt():
    assert synthesize_shortest_regex(["a", "b"], ["c", "aa"], "abc") == "a|b"


def test_synth_plus():
    assert synthesize_shortest_regex(["a", "aa", "aaa"], ["b", "ab", "ba"], "ab") == "a+"


def test_synth_optional():
    assert synthesize_shortest_regex(["a", "ab"], ["b", "abb"], "ab") == "ab?"


def test_regex_induction_label_scores_one():
    task = RegexInduction()
    p = task.generate()
    assert p is None or task.score_answer(p.answer, p) == 1.0


def test_regex_induction_rejects_forbidden_syntax():
    task = RegexInduction()
    p = task.generate()
    if p is not None:
        assert task.score_answer(".", p) == 0.0
        assert task.score_answer("[ab]", p) == 0.0


def test_regex_induction_label_not_hidden_artifact():
    task = RegexInduction()
    for _ in range(20):
        p = task.generate()
        if p is None:
            continue
        assert p.answer == p.metadata["shortest_regex"]
        assert "hidden_regex" in p.metadata


def test_regex_induction_prompt_examples_first():
    task = RegexInduction()
    p = task.generate()
    assert p is None or task.prompt(p.metadata).startswith("Positive:")


def test_regex_retrieval_label_scores_one():
    task = RegexRetrieval()
    p = task.generate()
    assert p is None or task.score_answer(p.answer, p) == 1.0


def test_regex_retrieval_prompt_and_order():
    task = RegexRetrieval()
    p = task.generate()
    if p is None:
        return
    prompt = task.prompt(p.metadata)
    assert prompt.startswith("Text:")
    assert "\nRegex:" in prompt
    assert "Return only a JSON array" in prompt
    assert json.loads(p.answer) == p.metadata["matches"]
    assert p.metadata["source"] in {"natural", "structured", "generated"}
