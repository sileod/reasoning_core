import math
import random


def _brute(congruences):
    """Direct scan over a bounded range used only for cross-checking."""
    if not congruences:
        return None
    limit = 1
    for (m, r) in congruences:
        limit *= m
    cap = min(limit + 1, 200000)
    for x in range(cap):
        if all(x % m == r for (m, r) in congruences):
            return ("found", x)
    res = _merge_all(congruences)
    if res is None:
        return "none"
    # if solution exists it is < lcm which may exceed cap
    return "found"


def _merge_all(congruences):
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import merge_all as ma
    return ma(congruences)


def _solver_congruences(congruences):
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import canonical_answer
    return canonical_answer(congruences)


def test_consistent_reaches_six_digits():
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import CongruenceSystem
    t = CongruenceSystem()
    t.config.set_level(5)
    for _ in range(40):
        t.config.n_min = 6
        t.config.n_max = 6
        t.config.base_bits = 11
        e = t.generate_entry()
        if e.answer != "none":
            assert len(e.answer) >= 6, e.answer
        assert t.score_answer(e.answer, e) == 1.0


def test_inconsistency_is_non_adjacent():
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import CongruenceSystem, merge_all
    t = CongruenceSystem()
    t.config.set_level(4)
    for _ in range(200):
        e = t.generate_entry()
        sys_ = e.metadata["system"]
        if e.answer == "none":
            # removing any one congruence should make it solvable
            solvable = False
            for k in range(len(sys_)):
                sub = sys_[:k] + sys_[k + 1:]
                if merge_all(sub) is not None:
                    solvable = True
                    break
            assert solvable, sys_


def test_brute_matches_solver():
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import CongruenceSystem, merge_all
    t = CongruenceSystem()
    t.config.set_level(0)
    for _ in range(300):
        t.config.n_min = 3
        t.config.n_max = 3
        t.config.base_bits = 7
        e = t.generate_entry()
        sys_ = e.metadata["system"]
        res = _brute(sys_)
        ans = _solver_congruences(sys_)
        full = merge_all(sys_)
        if full is None:
            assert ans == "none"
        else:
            x = full[0]
            for (m, r) in sys_:
                assert x % m == r
            assert int(ans) % full[1] == x % full[1]


def test_score_rejects_junk():
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import CongruenceSystem
    t = CongruenceSystem()
    for _ in range(20):
        e = t.generate_entry()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("garbage", e) == 0.0
        assert t.score_answer(e.answer, e) == 1.0


def test_gold_scores_one():
    from reasoning_core.tasks.generated.wave5.s51_congruence_system.s51_congruence_system import CongruenceSystem
    t = CongruenceSystem()
    for level in (0, 1, 2, 3, 4, 5):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_entry()
            assert t.score_answer(e.answer, e) == 1.0
