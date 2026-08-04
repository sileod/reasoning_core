import pytest

from reasoning_core.template import Entry, edict
from reasoning_core.tasks.binding import (
    LambdaReduction,
    RewriteSystem,
    Rule,
    UnificationEntailment,
    _alpha_normalize,
    _debruijn,
    _parse_lam,
    _pretty,
    _rw_parse,
    _rw_step,
    _subst,
)


@pytest.mark.parametrize("text", ["(a @ b)", "a @ garbage", "(\\x.x))"])
def test_lambda_parser_rejects_unconsumed_input(text):
    with pytest.raises(ValueError):
        _parse_lam(text)


@pytest.mark.parametrize("text", ["add(a,b) garbage", "add(a,@b)", "add(a,b))"])
def test_rewrite_parser_rejects_unconsumed_input(text):
    with pytest.raises(ValueError):
        _rw_parse(text)


def test_rewrite_parser_accepts_numeric_atoms_and_whitespace():
    assert _rw_parse(" add( 0, mul(2, 1) ) ") == ('add', '0', ('mul', '2', '1'))


def test_substitution_does_not_rename_when_variable_is_absent():
    term = ('l', 'y', ('v', 'z'))
    assert _subst(term, 'x', ('v', 'y')) == term


def test_alpha_normalization_is_canonical_and_preserves_free_names():
    a = ('l', 'z', ('l', 'q', ('a', ('v', 'z'), ('v', 'q'))))
    b = ('l', 'u', ('l', 'v', ('a', ('v', 'u'), ('v', 'v'))))
    assert _pretty(_alpha_normalize(a)) == _pretty(_alpha_normalize(b))
    capture_sensitive = ('l', 'z', ('a', ('v', 'z'), ('v', 'x0')))
    normalized = _alpha_normalize(capture_sensitive)
    assert _debruijn(normalized) == _debruijn(capture_sensitive)
    assert _debruijn(_parse_lam(_pretty(normalized))) == _debruijn(capture_sensitive)


def test_rewrite_step_uses_position_priority_before_rule_priority():
    rules = [
        Rule('inner', ('g', 'X'), 'X'),
        Rule('root', ('f', 'X'), 'a'),
    ]
    assert _rw_step(('f', ('g', 'b')), rules) == ('a', 'root')


def test_prompts_state_operational_semantics():
    prompt = RewriteSystem().render_prompt({'payload': {'rules': 'r', 'term': 't'}})
    assert 'position priority first; rule priority second' in prompt
    prompt = LambdaReduction().render_prompt({'term': r'(\x.x)'})
    assert 'compared up to α-equivalence' in prompt


def test_unification_entailment_uses_capitalized_boolean_answers():
    task = UnificationEntailment()
    entry = task.generate_example()

    assert entry.answer in {"Yes", "No"}
    assert "Answer Yes" in entry.prompt and "answer No" in entry.prompt
    assert task.score_answer(entry.answer.lower(), entry) == 1


def test_lambda_scoring_partially_rewards_beta_equivalent_non_normal_form():
    task = LambdaReduction()
    entry = Entry(metadata=edict(beta_steps=1), answer="a")

    assert task.score_answer("a", entry) == 1.0
    assert task.score_answer(r"((\x.x) a)", entry) == 0.8
    assert task.score_answer("b", entry) == 0.0
