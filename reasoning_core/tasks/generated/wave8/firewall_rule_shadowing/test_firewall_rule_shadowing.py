import random

from reasoning_core.tasks.generated.wave8.firewall_rule_shadowing.firewall_rule_shadowing import (
    FirewallRuleShadowing,
    FirewallRuleShadowingConfig,
    _contains,
)


def test_gold_scores_one():
    task = FirewallRuleShadowing()
    for _ in range(30):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_verify_gold_is_first_shadowed():
    task = FirewallRuleShadowing()
    for _ in range(30):
        ex = task.generate_example()
        rules = ex.metadata.payload["rules"]
        parsed = []
        for line in rules:
            parts = line.split()
            parsed.append((parts[0], parts[1], int(parts[2].split("-")[0]),
                           int(parts[2].split("-")[1]), int(parts[3].split("-")[0]),
                           int(parts[3].split("-")[1]), parts[4]))
        gold_parts = ex.answer.split()
        gold_index = int(gold_parts[0])
        found = None
        for i in range(len(parsed)):
            p2 = parsed[i][1]
            s2_lo, s2_hi = int(parsed[i][2]), int(parsed[i][3])
            d2_lo, d2_hi = int(parsed[i][4]), int(parsed[i][5])
            a2 = parsed[i][6]
            is_shadow = False
            for j in range(i):
                pj = parsed[j][1]
                jlo, jhi = int(parsed[j][2]), int(parsed[j][3])
                klo, khi = int(parsed[j][4]), int(parsed[j][5])
                b = parsed[j][6]
                if _contains(pj, jlo, jhi, klo, khi, p2, s2_lo, s2_hi, d2_lo, d2_hi) and b == a2:
                    is_shadow = True
                    break
            if is_shadow:
                found = i
                break
        assert found == gold_index


def test_junk_and_empty_score_zero():
    task = FirewallRuleShadowing()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("junk garbage", ex) == 0.0
    assert task.score_answer("None", ex) == 0.0


def test_difficulty_changes():
    n0 = FirewallRuleShadowingConfig.n_rules
    task = FirewallRuleShadowing()
    task.config.set_level(5)
    assert task.config.n_rules > n0
    assert task.config.addr_max > FirewallRuleShadowingConfig.addr_max


def test_distinct_answers():
    task = FirewallRuleShadowing()
    answers = {task.generate_example().answer for _ in range(40)}
    assert len(answers) >= 10
