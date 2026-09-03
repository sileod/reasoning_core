from collections import Counter
from itertools import chain

import pytest

from reasoning_core.tasks.math_metamath import (
    MetamathConfig,
    MetamathCoreSelect,
    MetamathEntailment,
    _closure,
    _database,
)


def _variable_shape(formula):
    db = _database()
    return sorted(Counter(t for t in formula if t in db.var_type).values())


def test_entailment_compares_token_balanced_premise_sets():
    task = MetamathEntailment(MetamathConfig())
    entry = task.generate_example(max_tokens=0)
    groups = [tuple(map(tuple, group)) for group in entry.metadata.raw_premise_sets]
    rules = [_database().rules[label] for label in entry.metadata.raw_rule_labels]
    target = tuple(entry.metadata.raw_conjecture)
    results = []
    for premises in groups:
        known = _closure(premises, rules, task.config.formula_len_cap, check_dv=False)
        results.append(known is not None and target in known)

    assert results.count(True) == 1
    assert entry.answer == "AB"[results.index(True)]
    assert Counter(chain.from_iterable(groups[0])) == Counter(chain.from_iterable(groups[1]))
    assert [_variable_shape(x) for x in groups[0]] == [_variable_shape(x) for x in groups[1]]
    assert "The answer is A or B." in entry.prompt
    assert MetamathEntailment.task_version == 2


@pytest.mark.parametrize("answer, gold, expected", [
    # Uppercasing the whole reply promoted the English article to a choice, and taking
    # the last standalone letter let the prose after the conclusion overrule it.
    ("C, which is a valid rule", "A", 0.0),
    ("C, which is a valid rule", "C", 1.0),
    # The original leniency: a bare [A-D] search matched the letter inside any word.
    ("reajrjrje9595!", "A", 0.0),
    # A reply that is nothing but a letter is a choice however it is cased.
    ("a", "A", 1.0),
    ("**C**", "C", 1.0),
    # Reasoning before concluding still settles on the last letter it wrote.
    ("Not A but D", "D", 1.0),
])
def test_core_select_scores_the_choice_not_a_letter_that_happens_to_appear(
    answer, gold, expected
):
    entry = type("Entry", (), {"answer": gold})()

    assert MetamathCoreSelect.score_answer(None, answer, entry) == expected
