from collections import Counter
from fractions import Fraction

import pytest

from reasoning_core.tasks.probabilistic_reasoning import (
    MostProbableEvidence,
    MostProbableEvidenceConfig,
    MostProbableOutcome,
    MostProbableOutcomeConfig,
    boolean_value,
    evidence_grammar,
    influential_atoms,
    mpe_answer,
)


def test_mpe_answer_rejects_ties():
    tied = "0.5::a.\n0.5::b.\nobserved :- (a;b).\nevidence(observed,true)."
    unique = "0.7::a.\n0.2::b.\nobserved :- (a;b).\nevidence(observed,true)."

    assert mpe_answer(tied) is None
    assert mpe_answer(unique) == '["a", "not b"]'


def test_negated_conjunction_is_not_rendered_as_unless():
    rule = next(
        rule for rule in evidence_grammar()._instances
        if rule.templates.get("problog") == "\\+{0}"
    )

    assert rule.templates["eng"].format("A") == "factor A is false"


def test_repeated_boolean_variables_are_checked_for_real_influence():
    formula = r"((a,b);(\+(a),c))"

    assert boolean_value(formula, {"a": True, "b": True, "c": False})
    assert set(influential_atoms(formula, ["a", "b", "c"])) == {"a", "b", "c"}


def test_mpe_richer_formulas_track_shared_influential_choices():
    config = MostProbableEvidenceConfig(min_shared_atoms=1, min_evidence_flips=1)
    task = MostProbableEvidence(config)
    entry = task.generate_entry()

    assert entry.metadata.shared_atom_count >= 1
    assert len(entry.metadata.influential_atoms) >= config.min_influential_atoms
    assert entry.metadata.evidence_flip_count >= config.min_evidence_flips
    assert task.score_answer(entry.answer, entry) == 1


def test_most_probable_outcome_is_stateless_and_batch_balanced():
    task = MostProbableOutcome()
    batch = task.generate_balanced_batch(batch_size=6)

    assert not hasattr(task, "_target_i")
    assert task.balancing_key_ratio == pytest.approx(1 / 3)
    assert Counter(problem.answer for problem in batch) == {"A": 2, "B": 2, "equal": 2}


def test_multistage_outcomes_use_exact_conditioned_probabilities():
    task = MostProbableOutcome(MostProbableOutcomeConfig(
        multistage_rate=1.0, observation_rate=1.0,
    ))

    for _ in range(12):
        entry = task.generate_entry()
        pa = Fraction(entry.metadata.probability_a)
        pb = Fraction(entry.metadata.probability_b)
        expected = "equal" if pa == pb else ("A" if pa > pb else "B")
        assert entry.metadata.mode == "multistage_exact"
        assert entry.metadata.n_draws >= 3
        assert entry.metadata.n_categories >= 3
        assert len(set(entry.metadata.replacements)) == 2
        assert entry.answer == expected
