import random


def test_roundtrip_and_scoring():
    random.seed(12345)
    from reasoning_core.tasks.generated.wave9.nfa_subset_construction.nfa_subset_construction import (
        NfaSubsetConstruction,
        NfaSubsetConstructionConfig,
    )

    conf = NfaSubsetConstructionConfig()
    task = NfaSubsetConstruction()
    task.config = conf

    for level in range(7):
        conf.set_level(level)
        for _ in range(20):
            e = task.generate_entry()
            assert task.score_answer(e.answer, e) == 1.0

    # junk / empty strings score less than 1
    assert task.score_answer("", e) < 1.0
    assert task.score_answer("abc", e) < 1.0
    assert task.score_answer(str(int(e.answer) + 1), e) < 1.0


def test_answer_is_domain_correct_positive_int():
    random.seed(999)
    from reasoning_core.tasks.generated.wave9.nfa_subset_construction.nfa_subset_construction import (
        NfaSubsetConstruction,
        NfaSubsetConstructionConfig,
    )

    conf = NfaSubsetConstructionConfig()
    task = NfaSubsetConstruction()
    task.config = conf
    for level in range(7):
        conf.set_level(level)
        for _ in range(50):
            e = task.generate_entry()
            v = int(e.answer)
            assert v >= 1
            assert v <= 2 ** conf.n_states


def test_determinism_same_seed():
    random.seed(42)
    from reasoning_core.tasks.generated.wave9.nfa_subset_construction.nfa_subset_construction import (
        NfaSubsetConstruction,
        NfaSubsetConstructionConfig,
    )

    task = NfaSubsetConstruction()
    task.config = NfaSubsetConstructionConfig()
    task.config.set_level(3)
    a = task.generate_entry().answer
    random.seed(42)
    b = task.generate_entry().answer
    assert a == b

