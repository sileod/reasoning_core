import random
import math


def _entropy(prob_list):
    h = 0.0
    total = sum(prob_list)
    for p in prob_list:
        if p > 0:
            q = p / total
            h -= q * math.log2(q)
    return h


def _expected_entropy(priors, partition):
    outcome_probs = [0.0] * len(partition)
    for i, p in enumerate(priors):
        for b, group in enumerate(partition):
            if i in group:
                outcome_probs[b] += p
                break
    expected = 0.0
    for b, op in enumerate(outcome_probs):
        if op <= 0:
            continue
        cond = [priors[i] / op for i in partition[b]]
        expected += op * _entropy(cond)
    return expected


def test_score_gold_and_junk():
    from reasoning_core.tasks.generated.wave9.diagnostic_test_selection.diagnostic_test_selection import (
        DiagnosticTestSelection,
    )

    task = DiagnosticTestSelection()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("xyz", x) == 0.0
    assert task.score_answer("42", x) == 0.0


def test_gold_is_verified_minimal():
    from reasoning_core.tasks.generated.wave9.diagnostic_test_selection.diagnostic_test_selection import (
        DiagnosticTestSelection,
    )

    task = DiagnosticTestSelection()
    x = task.generate_example()
    labels = sorted(task.config.n_tests and "ABCDEFGH"[:task.config.n_tests])
    rec = {k: _expected_entropy(x.metadata["priors"], p)
           for k, p in zip(labels, x.metadata["partition_blocks"])}
    best = min(rec, key=lambda k: (rec[k], k))
    assert x.answer == best


def test_expected_ambiguity_matches_callable():
    from reasoning_core.tasks.generated.wave9.diagnostic_test_selection.diagnostic_test_selection import (
        DiagnosticTestSelection,
    )

    task = DiagnosticTestSelection()
    x = task.generate_example()
    for k, p in zip("ABCDEFGH"[:task.config.n_tests], x.metadata["partition_blocks"]):
        assert abs(x.metadata["scores"][k] - _expected_entropy(x.metadata["priors"], p)) < 1e-9


def test_answer_is_single_letter():
    from reasoning_core.tasks.generated.wave9.diagnostic_test_selection.diagnostic_test_selection import (
        DiagnosticTestSelection,
    )

    task = DiagnosticTestSelection()
    for _ in range(20):
        x = task.generate_example()
        assert len(x.answer) == 1 and x.answer.isalpha()
