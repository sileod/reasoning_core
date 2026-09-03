import random

from reasoning_core.tasks.generated.wave8.finite_group_inverse.finite_group_inverse import (
    _alternating_table,
    _available_kinds,
    _build_group,
    _cyclic_table,
    _dihedral_table,
    _q8_table,
    _symmetric_table,
)
from reasoning_core.tasks.generated.wave8.finite_group_inverse.finite_group_inverse import (
    FiniteGroupInverse,
)


def _is_group(table, n):
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if table[table[a][b]][c] != table[a][table[b][c]]:
                    return False
    ids = [r for r in range(n) if all(table[r][c] == c and table[c][r] == c for c in range(n))]
    return len(ids) == 1


def _check_tables():
    cases = [
        (4, "cyclic", 4),
        (6, "dihedral", 6),
        (8, "dihedral", 8),
        (8, "q8", 8),
        (6, "s3", 6),
        (12, "a4", 12),
        (24, "s4", 24),
        (5, "cyclic", 5),
        (9, "cyclic", 9),
    ]
    for n, kind, expect in cases:
        t = _build_group(n, kind)
        assert t is not None, kind
        assert len(t) == expect, kind
        assert _is_group(t, expect), kind


def test_all_group_tables_are_groups():
    _check_tables()


def test_every_order_has_cyclic():
    for n in range(4, 25):
        assert "cyclic" in _available_kinds(n)


def test_gold_answer_scores_one_across_levels():
    task = FiniteGroupInverse()
    for level in (0, 2, 5):
        task.config.set_level(level)
        random.seed(level * 1000 + 7)
        for _ in range(3):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0
            assert isinstance(e.answer, str) and e.answer.isalpha()


def test_inverse_is_actually_inverse():
    task = FiniteGroupInverse()
    random.seed(42)
    for _ in range(20):
        e = task.generate_example()
        labels = [chr(65 + i) for i in range(e.metadata.order)]
        table_rows = e.metadata.payload["Cayley table"].splitlines()
        headers = table_rows[0].split()
        identity = e.metadata.identity
        element = e.metadata.element
        answer = e.metadata.inverse
        assert answer == e.answer
        assert element in headers and answer in headers
        ri = headers.index(element)
        ci = headers.index(answer)
        assert table_rows[ri + 1].split()[ci + 1] == identity
