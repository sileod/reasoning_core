import random
from math import gcd

from reasoning_core.tasks.generated.wave4.s45_equation_balancing.equation_balancing import (
    EquationBalancing, EquationBalancingConfig, _conserves, _parse_formula)


def _signs(n_species, n_react):
    return [1] * n_react + [-1] * (n_species - n_react)


def test_gold_scores_one():
    for seed in range(20):
        random.seed(seed)
        task = EquationBalancing()
        task.config.set_level(0)
        entry = task.generate_entry()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_conservation_holds():
    for seed in range(20):
        random.seed(seed)
        task = EquationBalancing()
        task.config.set_level(0)
        entry = task.generate_entry()
        coeffs = [int(c) for c in entry.answer.split(",")]
        n_species = len(coeffs)
        n_react = len(entry.metadata.reactants)
        species = [{"formula": f, "elems": _parse_formula(f)} for f in entry.metadata.species]
        assert _conserves(species, _signs(n_species, n_react), coeffs)


def test_answer_is_smallest_positive():
    for seed in range(20):
        random.seed(seed)
        task = EquationBalancing()
        task.config.set_level(0)
        entry = task.generate_entry()
        coeffs = [int(c) for c in entry.answer.split(",")]
        g = 0
        for v in coeffs:
            g = gcd(g, v)
        assert g == 1
        assert all(c > 0 for c in coeffs)
        assert len(set(coeffs)) >= 2


def test_wrong_answers_score_zero():
    random.seed(0)
    task = EquationBalancing()
    task.config.set_level(0)
    entry = task.generate_entry()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("junk", entry) == 0.0
    assert task.score_answer("0,0,0,0", entry) == 0.0
    assert task.score_answer("1", entry) == 0.0


def test_difficulty_changes():
    cfg = EquationBalancingConfig()
    cfg.set_level(0)
    l0 = cfg.get_true_value('n_species')
    cfg.set_level(5)
    assert cfg.get_true_value('n_species') >= l0


def test_all_levels_generate():
    for level in range(7):
        random.seed(100 + level)
        task = EquationBalancing()
        task.config.set_level(level)
        entry = task.generate_entry()
        assert entry.answer != ""
        assert task.score_answer(entry.answer, entry) == 1.0


def test_answer_varies():
    random.seed(0)
    answers = set()
    for _ in range(30):
        task = EquationBalancing()
        task.config.set_level(0)
        answers.add(tuple(task.generate_entry().answer.split(",")))
    assert len(answers) > 1
