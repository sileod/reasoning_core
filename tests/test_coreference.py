import json

import pytest
import z3

import reasoning_core.tasks.coreference as coreference_module
from reasoning_core.tasks.coreference import (
    Coreference,
    CoreferenceConfig,
    Factor,
    compile_z3,
    entailment_status,
    propagation_depth,
    propagation_round,
)


def test_permutation_propagates_all_positions_synchronously():
    domains = {
        0: {0}, 1: {1}, 2: {2},
        3: {0, 1, 2}, 4: {0, 1, 2}, 5: {0, 1, 2},
    }
    factor = Factor('permutation', (3, 4, 5, 0, 1, 2), (2, 0, 1))
    updated = propagation_round(domains, [factor])
    assert updated[3] == {2}
    assert updated[4] == {0}
    assert updated[5] == {1}


def test_all_different_performs_width_three_elimination():
    domains = {0: {0}, 1: {1}, 2: {0, 1, 2}}
    factor = Factor('all_different', (0, 1, 2))
    assert propagation_round(domains, [factor])[2] == {2}


def test_z3_entailment_is_tri_state(monkeypatch):
    class UnknownSolver:
        def set(self, **kwargs):
            pass

        def add(self, *constraints):
            pass

        def check(self):
            return z3.unknown

    monkeypatch.setattr(coreference_module.z3, 'Solver', UnknownSolver)
    assert entailment_status(0, 0, {0: {0, 1}}, []) is None


@pytest.mark.parametrize(
    ('family', 'depths'),
    [
        ('permutation_forward', (1, 3, 6)),
        ('permutation_backward', (1, 3, 6)),
        ('branching_elimination', (2, 3, 6)),
    ],
)
def test_families_certify_depth_and_support(family, depths):
    for depth in depths:
        task = Coreference(CoreferenceConfig(
            family=family, target_depth=depth, max_attempts=30))
        entry = task.generate_entry()
        metadata = entry.metadata
        assert metadata.propagation_depth == depth
        assert metadata.support_factor_indices == metadata.necessary_factor_indices
        assert metadata.full_csp_sat and metadata.z3_entailed
        assert task.score_answer(entry.answer, entry) == 1
        query = str(metadata.query_idx)
        assert len(metadata.propagation_trace[-2][query]) > 1
        assert metadata.propagation_trace[-1][query] == [metadata.answer_eid]


@pytest.mark.parametrize('width', [3, 4, 6, 8])
def test_variable_width_answer_domain(width):
    entry = Coreference(CoreferenceConfig(
        family='permutation_forward', n_entities=width,
        target_depth=2)).generate_entry()
    assert entry.metadata.width == width
    assert len(entry.metadata.answer_domain_eids) == width


def test_backward_family_requires_later_evidence():
    entry = Coreference(CoreferenceConfig(
        family='permutation_backward')).generate_entry()
    assert entry.metadata.prefix_entailed is False
    assert any(factor['sentence'] >= entry.metadata.query_sentence
               for factor in entry.metadata.factors)


def test_branching_family_uses_both_branches_and_elimination():
    entry = Coreference(CoreferenceConfig(
        family='branching_elimination', target_depth=4,
        n_entities=5)).generate_entry()
    kinds = [entry.metadata.factors[index]['kind']
             for index in entry.metadata.support_factor_indices]
    assert kinds.count('permutation') >= 2
    assert kinds.count('equal') == 4
    assert 'all_different' in kinds


def test_distractors_never_name_the_answer():
    task = Coreference(CoreferenceConfig(n_distractor_sentences=4))
    entry = task.generate_entry()
    distractors = entry.metadata.sentences.splitlines()[-4:]
    assert all(entry.answer not in line for line in distractors)


def test_prompt_is_concise_and_explicit():
    task = Coreference(CoreferenceConfig(family='permutation_backward'))
    entry = task.generate_entry()
    prompt = task.render_prompt(entry.metadata)
    assert 'source positions' in prompt
    assert 'later statements' in prompt
    assert len(prompt.split()) < 180


def test_generate_example_and_wrong_answer_scoring():
    task = Coreference(CoreferenceConfig(family='mixed'))
    entry = task.generate_example()
    assert task.score_answer(entry.answer, entry) == 1
    assert task.score_answer('DefinitelyWrong', entry) == 0


def test_metadata_is_strictly_json_serializable():
    entry = Coreference().generate_entry()
    assert json.loads(json.dumps(dict(entry.metadata)))['answer_eid'] == entry.metadata.answer_eid


def test_compile_z3_rejects_duplicate_all_different_values():
    domains = {0: {0}, 1: {0}}
    _, constraints = compile_z3(
        domains, [Factor('all_different', (0, 1))])
    solver = z3.Solver()
    solver.add(*constraints)
    assert solver.check() == z3.unsat


def test_branching_depth_one_fails_clearly():
    with pytest.raises(ValueError, match='requires target_depth'):
        Coreference(CoreferenceConfig(
            family='branching_elimination', target_depth=1)).generate_entry()
