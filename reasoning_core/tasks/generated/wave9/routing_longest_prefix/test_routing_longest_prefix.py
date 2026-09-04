import random

from reasoning_core.tasks.generated.wave9.routing_longest_prefix.routing_longest_prefix import (
    RoutingLongestPrefix,
    RoutingLongestPrefixConfig,
    _dst,
    _fmt_prefix,
    _matches,
    _select_hop,
)


def _sorted_entries(entry):
    bits = int(entry.metadata.bits)
    return [{
        "prefix": int(e["prefix"], 2) << (bits - e["plen"]),
        "plen": int(e["plen"]),
        "hop": e["hop"],
        "metric": int(e["metric"]),
    } for e in entry.metadata.table]


def test_gold_scores_one():
    task = RoutingLongestPrefix()
    for _ in range(50):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_wrong_and_garbage_not_full():
    task = RoutingLongestPrefix()
    for _ in range(30):
        x = task.generate_example()
        assert task.score_answer("", x) < 1.0
        assert task.score_answer("garbage", x) < 1.0
        assert task.score_answer(None, x) < 1.0


def test_distractors_exclude_correct():
    task = RoutingLongestPrefix()
    for _ in range(30):
        x = task.generate_example()
        for d in task.distractor_candidates(x):
            assert d != x.answer
            assert task.score_answer(d, x) < 1.0


def test_answer_matches_algorithm():
    task = RoutingLongestPrefix()
    for _ in range(50):
        x = task.generate_example()
        addr = int(x.metadata.destination, 2)
        bits = int(x.metadata.bits)
        winner = _select_hop(bits, addr, _sorted_entries(x))
        assert winner == x.answer


def test_difficulty_changes_config():
    cfg = RoutingLongestPrefixConfig()
    base = (cfg.bits, cfg.n_entries, cfg.n_hops, cfg.max_metric)
    cfg.set_level(6)
    hi = (cfg.bits, cfg.n_entries, cfg.n_hops, cfg.max_metric)
    assert hi != base


def test_metadata_json_roundtrip():
    import json

    task = RoutingLongestPrefix()
    x = task.generate_example()
    json.dumps(dict(x.metadata))


def test_every_level_generates():
    task = RoutingLongestPrefix()
    for level in range(7):
        task.config.set_level(level)
        x = task.generate_example()
        assert x.answer is not None
        assert task.score_answer(x.answer, x) == 1.0


def test_match_consistency():
    bits = 8
    addr = 0b10110110
    for plen in range(1, 9):
        prefix = (addr >> (bits - plen)) << (bits - plen) if plen > 0 else 0
        assert _matches(bits, prefix, plen, addr)


def test_nonmatch_flip_msb():
    bits = 8
    addr = 0b00110011
    flipped = addr ^ (1 << (bits - 1))
    for plen in range(1, 9):
        prefix = (flipped >> (bits - plen)) << (bits - plen)
        assert not _matches(bits, prefix, plen, addr)


def test_answer_carries_witness_not_surface():
    task = RoutingLongestPrefix()
    for _ in range(20):
        x = task.generate_example()
        t = task.render_prompt(x.metadata)
        assert x.answer not in x.metadata.destination
        assert x.answer != t.strip().split()[-1]


def _generate_levels(seed):
    random.seed(seed)
    task = RoutingLongestPrefix()
    out = []
    for level in (0, 2, 5):
        task.config.seed = seed
        task.config.set_level(level)
        x = task.generate_example()
        out.append((level, task.render_prompt(x.metadata), x.answer))
    return out


def test_deterministic_generation_with_fixed_seed():
    a = _generate_levels(265960371)
    b = _generate_levels(265960371)
    assert a == b


def test_answer_is_a_hop_in_table():
    task = RoutingLongestPrefix()
    for _ in range(40):
        x = task.generate_example()
        hops = {e["hop"] for e in x.metadata.table}
        assert x.answer in hops


def test_prompt_self_contained():
    task = RoutingLongestPrefix()
    x = task.generate_example()
    t = task.render_prompt(x.metadata)
    assert x.metadata.destination in t
    assert "longest prefix" in t.lower()
    assert "lowest metric" in t.lower()
    assert "lexicographically" in t.lower()


def test_matches_domain_consistent():
    task = RoutingLongestPrefix()
    for _ in range(40):
        x = task.generate_example()
        assert int(x.metadata.bits) > 0
        assert len(x.metadata.destination) == int(x.metadata.bits)
        for e in x.metadata.table:
            assert 1 <= int(e["plen"]) <= int(x.metadata.bits)
            assert len(e["prefix"]) == int(e["plen"])
            assert int(e["metric"]) >= 0

