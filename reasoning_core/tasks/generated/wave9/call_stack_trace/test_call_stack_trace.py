import random

from reasoning_core.tasks.generated.wave9.call_stack_trace.call_stack_trace import (
    CallStackTrace, CallStackConfig)


def test_roundtrip_scores_one():
    random.seed(1)
    t = CallStackTrace()
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    random.seed(2)
    t = CallStackTrace()
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("abc", e) == 0.0
    assert t.score_answer("x", e) == 0.0


def test_all_levels_generate():
    for level in range(7):
        cfg = CallStackConfig()
        cfg.set_level(level)
        t = CallStackTrace(config=cfg)
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_target_subtree_truth():
    random.seed(3)
    t = CallStackTrace()
    for _ in range(20):
        e = t.generate_example()
        total = _subtree_sum_from_repr(e.metadata['tree'], e.metadata['target'])
        assert total == int(e.answer)


def _build(s, i, nodes):
    neg = False
    if s[i] == '-':
        neg = True
        i += 1
    num = ""
    while i < len(s) and s[i].isdigit():
        num += s[i]
        i += 1
    val = -int(num) if neg else int(num)
    idx = len(nodes)
    nodes.append({'val': val, 'children': []})
    if i < len(s) and s[i] == '(':
        i += 1
        while s[i] != ')':
            i, cidx = _build(s, i, nodes)
            nodes[idx]['children'].append(cidx)
            if s[i] == ',':
                i += 1
        i += 1
    return i, idx


def _subtree_sum_from_repr(s, target):
    nodes = []
    _build(s, 0, nodes)
    sums = [0] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        total = nodes[idx]['val']
        for c in nodes[idx]['children']:
            total += sums[c]
        sums[idx] = total
    return sums[target]
