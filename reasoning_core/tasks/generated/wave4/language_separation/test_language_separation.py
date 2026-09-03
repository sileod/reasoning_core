from reasoning_core.tasks.generated.wave4.s38_language_separation.language_separation import (
    LanguageSeparation,
    _build_one,
    _shortest_witness,
)


def _witness_len(task, entry):
    m = entry.metadata
    if entry.answer == "none":
        return None
    return len(entry.answer)


def test_generate_and_score():
    t = LanguageSeparation()
    for _ in range(40):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_shortest_witness_sanity():
    import random
    alphabet = ['a', 'b']
    s1, acc1, t1 = _build_one(4, alphabet, random)
    s2, acc2, t2 = _build_one(4, alphabet, random)
    w = _shortest_witness(alphabet, s1, acc1, t1, s2, acc2, t2)
    if w is not None:
        # verify it is accepted by 1, rejected by 2
        p = s1
        for c in w:
            p = t1[p][alphabet.index(c)]
        q = s2
        for c in w:
            q = t2[q][alphabet.index(c)]
        assert p in acc1
        assert q not in acc2


def test_none_when_identical():
    import random
    s1, acc1, t1 = _build_one(3, ['a', 'b'], random)
    # build identical second machine -> no witness
    w = _shortest_witness(['a', 'b'], s1, acc1, t1, s1, set(acc1), t1)
    assert w is None


def test_difficulty_changes():
    t = LanguageSeparation()
    t.config.set_level(0)
    n0 = int(t.config.n_states)
    t.config.set_level(5)
    n5 = int(t.config.n_states)
    assert n5 > n0
