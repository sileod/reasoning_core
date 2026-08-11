from easydict import EasyDict as edict
from nltk import CFG

from reasoning_core.tasks.grammar import (
    GrammarConfig,
    ParsingDerivation,
    grammar_text,
    labeled_rules,
)
from reasoning_core.template import Entry


def test_parsing_derivation_soft_score():
    entry = Entry(edict(labeled_g="R0: S -> A B\nR1: A -> 'a'\nR2: B -> 'b'"), "R0 R1 R2")
    task = ParsingDerivation()

    assert task.score_answer("Rules: R0, R1 R2", entry) == 1.0
    assert 0.0 < task.score_answer("R0 R2", entry) < 1.0
    assert task.score_answer("R8 R9", entry) == 0.0


def test_parsing_derivation_uses_shorter_defaults():
    config = ParsingDerivation().config
    assert (config.target_num_rules, config.min_prod_depth, config.max_prod_depth, config.max_tokens) == (8, 3, 5, 12)


def test_grammar_rendering_supports_configured_bnf_operator():
    grammar = CFG.fromstring("S -> A\nA -> 'a'")

    arrow = grammar_text(grammar, GrammarConfig(bnf_operator_prob=0))
    definition = grammar_text(grammar, GrammarConfig(bnf_operator_prob=1))

    assert " -> " in arrow and " ::= " not in arrow
    assert " ::= " in definition and " -> " not in definition
    assert GrammarConfig().bnf_operator_prob == 0.5


def test_labeled_rules_map_bnf_style_back_to_productions():
    rendered, labels = labeled_rules(edict(g="S ::= A\nA ::= 'a'"))

    assert "::=" in rendered
    assert set(labels) == {"S -> A", "A -> 'a'"}
