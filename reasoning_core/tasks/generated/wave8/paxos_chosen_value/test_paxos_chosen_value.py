import random

from reasoning_core.tasks.generated.wave8.paxos_chosen_value.paxos_chosen_value import (
    PaxosChosenValue, PaxosChosenValueConfig, _quorum, TASK_META
)


def test_roundtrip():
    t = PaxosChosenValue()
    for level in (0, 3, 6):
        cfg = PaxosChosenValueConfig()
        cfg.apply_difficulty(level)
        t.config = cfg
        random.seed(level)
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_score_garbage():
    t = PaxosChosenValue()
    cfg = PaxosChosenValueConfig()
    t.config = cfg
    random.seed(0)
    e = t.generate_example()
    assert t.score_answer("", e) < 1.0
    assert t.score_answer("garbage", e) < 1.0


def test_meta():
    assert TASK_META["hypothesis"] == "W1-041"
    assert TASK_META["parent_source_id"] is None


def test_quorum():
    assert _quorum(3) == 2
    assert _quorum(4) == 3
    assert _quorum(5) == 3


def _parse(meta):
    quorum = meta.quorum
    votes = []
    import re
    for line in meta.proposal_lines:
        m = re.match(r"Acceptor (\d+) voted for proposal (\d+) with value (-?\d+)\.", line)
        votes.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return quorum, votes


def test_chosen_highest_quorum_value():
    t = PaxosChosenValue()
    for level in (0, 3, 6):
        cfg = PaxosChosenValueConfig()
        cfg.apply_difficulty(level)
        t.config = cfg
        for _ in range(50):
            e = t.generate_example()
            q, votes = _parse(e.metadata)
            by_prop = {}
            for a, p, v in votes:
                by_prop.setdefault(p, []).append(v)
            if e.answer == "None":
                assert all(len(v) < q for v in by_prop.values())
            else:
                ans = int(e.answer)
                quorum_props = [p for p, v in by_prop.items() if len(v) >= q]
                assert quorum_props, "must have at least one quorum proposal"
                highest = max(quorum_props)
                assert all(x == ans for x in by_prop[highest])
                assert ans >= 0

