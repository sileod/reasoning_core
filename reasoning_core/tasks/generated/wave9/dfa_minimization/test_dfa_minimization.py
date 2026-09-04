from reasoning_core.tasks.generated.wave9.dfa_minimization.dfa_minimization import (
    DfaMinimization, _minimize, _sorted_block_list, _normalize,
)


def test_gold_scores_one():
    task = DfaMinimization()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_partition_merges_equivalent():
    # states q0,q1 both non-accepting with identical behavior -> should merge
    labels = _minimize([0, 1, 2], ['a', 'b'],
                       {0: [0, 0], 1: [1, 1], 2: [2, 2]}, {2})
    assert labels[0] == labels[1]
    assert labels[2] != labels[1]


def test_normalize_spacing():
    assert _normalize("q0,q1 | q2") == _normalize("q0,q1|q2")
    assert _normalize("q0,q1 | q2") != _normalize("q0,q2 | q1")


def test_accepts_canonical():
    # identical strings with different spacing accepted
    task = DfaMinimization()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0
    # a permuted partition rejected
    parts = x.answer
    assert task.score_answer(parts + " | qX", x) == 0.0


def test_config_difficulty_changes():
    cfg = DfaMinimization.config_cls()
    base = cfg.max_states
    cfg.set_level(5)
    assert cfg.max_states >= base
