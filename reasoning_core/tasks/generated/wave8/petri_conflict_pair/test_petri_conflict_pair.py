import random

from reasoning_core.tasks.generated.wave8.petri_conflict_pair.petri_conflict_pair import (
    PetriConflictPair, PetriConflictPairConfig, parse_answer, conflicting_pairs, enabled,
)


def test_gold_scores_one():
    task = PetriConflictPair()
    for _ in range(40):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_answer_exists_and_parseable():
    task = PetriConflictPair()
    e = task.generate_example()
    assert e.answer == 'None' or parse_answer(e.answer) is not None


def test_junk_scores_zero():
    task = PetriConflictPair()
    e = task.generate_example()
    assert task.score_answer('', e) == 0.0
    assert task.score_answer('garbage_input_here', e) == 0.0


def test_both_answer_modes_produced():
    task = PetriConflictPair()
    answers = {task.generate_example().answer for _ in range(60)}
    assert 'None' in answers
    non_none = [a for a in answers if a != 'None']
    assert len(non_none) >= 2


def test_conflict_verifier():
    random.seed(7)
    task = PetriConflictPair()
    for _ in range(50):
        e = task.generate_example()
        pt = {p['name']: p['tokens'] for p in e.metadata.places}
        trans = [{'name': t['name'], 'inputs': [(a, b) for (a, b) in t['inputs']]}
                 for t in e.metadata.transitions]
        found = conflicting_pairs(trans, pt)
        if e.answer == 'None':
            assert not found
        else:
            assert found


def test_difficulty_changes():
    cfg = PetriConflictPairConfig()
    a = cfg.n_places
    cfg.set_level(5)
    b = cfg.n_places
    assert b > a


def test_wrong_pair_rejected():
    task = PetriConflictPair()
    for _ in range(40):
        e = task.generate_example()
        if e.answer == 'None':
            assert task.score_answer('t9,t8,p7,1', e) == 0.0
        else:
            assert task.score_answer('t99,t98,p97,1', e) == 0.0


def test_deficit_positive():
    task = PetriConflictPair()
    for _ in range(60):
        e = task.generate_example()
        if e.answer != 'None':
            parts = e.answer.split(',')
            assert int(parts[3]) >= 1


def test_prompt_not_surface_readable():
    task = PetriConflictPair()
    for _ in range(30):
        e = task.generate_example()
        prom = task.render_prompt(e.metadata)
        last = prom.split()[-1].strip('(),.')
        assert last != e.answer
        assert e.answer not in (prom.split('\n')[0], )
