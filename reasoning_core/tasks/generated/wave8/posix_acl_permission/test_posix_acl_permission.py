import random

from reasoning_core.tasks.generated.wave8.posix_acl_permission.posix_acl_permission import (
    PosixAclPermission,
    effective_perm,
    PERM_CHARS,
)

random.seed(1)


def _fresh():
    return PosixAclPermission()


def test_summary_present():
    s = PosixAclPermission.summary
    assert isinstance(s, str) and s.strip()
    assert "\n" not in s


def test_example_scores_one_at_all_levels():
    t = _fresh()
    for level in (0, 2, 5):
        t.config.set_level(level)
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1


def test_answer_format_valid():
    t = _fresh()
    for _ in range(30):
        ex = t.generate_example()
        parts = ex.answer.split(";")
        assert len(parts) == len(ex.metadata.subjects)
        for p in parts:
            assert set(p) <= set(PERM_CHARS) or p == "-"


def test_junk_scores_zero():
    t = _fresh()
    ex = t.generate_example()
    for bad in ("", " ", "reajrjrje9595!", "rwx"):
        assert t.score_answer(bad, ex) < 1


def test_difficulty_changes():
    t = _fresh()
    base = t.config.set_level(0) or t.config
    n0 = base.n_subjects
    t.config.set_level(6)
    assert t.config.n_subjects >= n0


def test_effective_matches_generator():
    t = _fresh()
    for _ in range(30):
        ex = t.generate_example()
        m = ex.metadata
        for s, groups in m.subjects:
            subj = s
            got = effective_perm(
                m.owner, m.owner_perm, m.named_users, m.owning_group,
                m.owngroup_perm, m.named_groups, sorted(m.named_groups),
                m.mask, m.other, subj, set(groups))
            idx = [x for x, _ in m.subjects].index(subj)
            assert got == m.perms[idx], (got, m.perms[idx])


def test_metadata_json_roundtrip_and_payload():
    import json
    t = _fresh()
    ex = t.generate_example()
    dumped = json.dumps(dict(ex.metadata))
    assert dict(ex.metadata)["_config"] and "payload" in ex.metadata


def test_subjects_distinctness_and_priority_branches():
    t = _fresh()
    seen_owner_other = False
    for _ in range(40):
        ex = t.generate_example()
        m = ex.metadata
        names = [s for s, _ in m.subjects]
        assert len(names) == len(set(names))
        assert m.owner not in names
        for s, groups in m.subjects:
            if s not in m.named_users and m.owning_group not in groups \
               and not any(g in groups for g in m.named_groups):
                assert effective_perm(m.owner, m.owner_perm, m.named_users,
                                      m.owning_group, m.owngroup_perm, m.named_groups,
                                      sorted(m.named_groups), m.mask, m.other,
                                      s, set(groups)) == m.other
                seen_owner_other = True
    assert seen_owner_other


def test_all_levels_generate():
    t = _fresh()
    for level in range(7):
        ex = t.generate_example(level=level)
        assert t.score_answer(ex.answer, ex) == 1
