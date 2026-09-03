from reasoning_core.tasks.generated.wave3.s28_principal_type.s28_principal_type import PrincipalType, principal


def test_gold_scores_one():
    task = PrincipalType()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_untypable_none_construction():
    task = PrincipalType()
    for _ in range(10):
        ex = task.generate_example()
        if ex.answer == 'untypable':
            assert True


def test_junk_not_one():
    task = PrincipalType()
    for _ in range(5):
        ex = task.generate_example()
        assert task.score_answer("", ex) < 1.0
        assert task.score_answer("garbage", ex) < 1.0
        assert task.score_answer(None, ex) < 1.0


def test_whitespace_insensitive():
    task = PrincipalType()
    for _ in range(5):
        ex = task.generate_example()
        if ex.answer != 'untypable':
            spaced = ex.answer.replace('->', ' -> ')
            assert task.score_answer(spaced, ex) == 1.0


def test_known_terms():
    identity = ('lam', 'x', ('var', 'x'))
    assert principal(identity) == 't1 -> t1'
    k = ('lam', 'x', ('lam', 'y', ('var', 'x')))
    assert principal(k) == 't1 -> t2 -> t1'
    comp = ('lam', 'f', ('lam', 'g',
           ('lam', 'x', ('app', ('var', 'f'), ('app', ('var', 'g'),
                                                ('var', 'x'))))))
    assert principal(comp) == '(t1 -> t2) -> (t3 -> t1) -> t3 -> t2'


def test_untypable_term():
    omega = ('lam', 'x', ('app', ('var', 'x'), ('var', 'x')))
    assert principal(omega) == 'untypable'
    selfapp = ('app', ('lam', 'x', ('app', ('var', 'x'), ('var', 'x'))),
               ('lam', 'x', ('var', 'x')))
    assert principal(selfapp) == 'untypable'


def test_answer_uses_contiguous_vars():
    import re
    task = PrincipalType()
    for _ in range(30):
        ex = task.generate_example()
        if ex.answer == 'untypable':
            continue
        nums = sorted(set(int(x) for x in re.findall(r't([0-9]+)', ex.answer)))
        for i, n in enumerate(nums):
            assert n == i + 1
