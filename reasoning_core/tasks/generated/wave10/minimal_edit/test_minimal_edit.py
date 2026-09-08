import random

from reasoning_core.template import stochastic_rounding as sround
from reasoning_core.tasks.generated.wave10.minimal_edit.minimal_edit import (
    MinimalEdit, MinimalEditConfig, normalize, _assemble_positive, _assemble_negative,
    _build_parts,
)


def test_gold_scores_one_each_level():
    task = MinimalEdit()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0
            assert e.answer != e.metadata.sentence


def test_junk_scores_zero():
    task = MinimalEdit()
    task.config.set_level(1)
    for _ in range(30):
        e = task.generate_entry()
        assert task.score_answer("", e) < 1.0
        assert task.score_answer("the chef bakes the cake", e) < 1.0
        assert task.score_answer("nonsense here", e) < 1.0


def test_source_and_answer_are_polarity_opposites():
    task = MinimalEdit()
    task.config.set_level(4)
    for _ in range(30):
        e = task.generate_entry()
        assert "does not" in e.metadata.sentence and "does not" not in e.answer \
            or ("does not" not in e.metadata.sentence and "does not" in e.answer)


def test_builders_reconstruct_each_other():
    random.seed(1234)
    cfg = MinimalEditConfig()
    cfg.set_level(5)
    for _ in range(30):
        subj, verb, obj, mods, adj = _build_parts(cfg)
        pos = _assemble_positive(subj, verb, obj, list(mods), adj)
        neg = _assemble_negative(subj, verb, obj, list(mods), adj)
        assert pos != neg
        assert pos.startswith(subj)
        assert neg.startswith(subj)


def test_normalize_apostrophe_equivalence():
    assert normalize("The chef does not bake the cake.") == normalize(
        "The chef doesn't bake the cake.")


def test_difficulty_changes_config():
    c = MinimalEditConfig()
    c0 = c.n_mods
    c.set_level(5)
    assert c.n_mods >= c0


def test_answer_varies_across_examples():
    task = MinimalEdit()
    task.config.set_level(6)
    answers = {task.generate_entry().answer for _ in range(40)}
    assert len(answers) > 25


def test_relative_clause_preserved_and_well_formed():
    task = MinimalEdit()
    task.config.set_level(6)
    for _ in range(30):
        e = task.generate_entry()
        if "who " not in e.metadata.sentence:
            continue
        src = e.metadata.sentence
        ans = e.answer
        assert ", who " in src
        assert src.count(", who ") == 1
        rel = src[src.index(", who ") + 2:src.index(", who ") + 2 +
                    src[src.index(", who "):].index(", ")]
        assert rel in ans
        # negated form must carry 'does not'; positive form must not
        assert ("does not" in src and "does not" not in ans) or \
               ("does not" not in src and "does not" in ans)


def test_modifiers_grow_with_level():
    task = MinimalEdit()
    lens = []
    for level in (0, 2, 4, 6):
        task.config.set_level(level)
        e = task.generate_entry()
        lens.append(len(e.answer))
    for a, b in zip(lens, lens[1:]):
        assert b > a
