from collections import Counter
from itertools import chain

from reasoning_core.tasks.math_metamath import (
    MetamathConfig,
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
