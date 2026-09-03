from reasoning_core.tasks.generated.wave3.s30_resistor_networks.s30_resistor_networks import (
    ResistorNetworks,
    ResistorNetworkConfig,
    evaluate,
    build,
    valid_tree,
    total_resistors,
)
from fractions import Fraction


def test_generate_scores_one():
    task = ResistorNetworks()
    entry = task.generate_example()
    assert task.score_answer(entry.answer, entry) == 1.0


def test_answer_fraction_positive():
    task = ResistorNetworks()
    entry = task.generate_example()
    frac = Fraction(entry.answer)
    assert frac > 0


def test_levels_scored():
    task = ResistorNetworks()
    for level in range(7):
        config = ResistorNetworkConfig()
        config.set_level(level)
        task.config = config
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_junk_scores_zero():
    task = ResistorNetworks()
    entry = task.generate_example()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("junk", entry) == 0.0


def test_equivalent_fractions_score_one():
    task = ResistorNetworks()
    entry = task.generate_example()
    frac = Fraction(entry.answer)
    assert task.score_answer(f"{frac.numerator * 2}/{frac.denominator * 2}", entry) == 1.0
