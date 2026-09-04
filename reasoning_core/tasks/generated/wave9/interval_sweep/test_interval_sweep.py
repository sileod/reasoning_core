import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import random
import reasoning_core.tasks.generated.wave9.interval_sweep.interval_sweep as mod


def test_generate_and_score_all_levels():
    random.seed(1)
    for level in range(7):
        t = mod.IntervalSweep()
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_entry()
            assert mod.IntervalSweep().score_answer(e.answer, e) == 1.0
            assert mod.IntervalSweep().score_answer("", e) == 0.0
            assert mod.IntervalSweep().score_answer("zzzjunk", e) == 0.0


def test_gold_reproduces_peak():
    t = mod.IntervalSweep()
    t.config.set_level(0)
    for _ in range(30):
        e = t.generate_entry()
        if e.metadata.mode != "peak":
            continue
        peak = int(e.answer)
        events = []
        for a, b in e.metadata.intervals:
            events.append((a, 1))
            events.append((b, -1))
        events.sort()
        cur = maxc = 0
        for _, d in events:
            cur += d
            maxc = max(maxc, cur)
        assert peak == maxc and peak >= 1


def test_gold_reproduces_merge():
    t = mod.IntervalSweep()
    t.config.set_level(3)
    for _ in range(30):
        e = t.generate_entry()
        if e.metadata.mode != "merge":
            continue
        recomputed = mod._merge(e.metadata.intervals)
        fmt = "; ".join(f"{a}-{b}" for a, b in recomputed)
        assert fmt == e.answer


def test_prompt_determines_merge_answer():
    ast_merge = mod._merge
    t = mod.IntervalSweep()
    t.config.set_level(3)
    seen = {}
    for _ in range(50):
        e = t.generate_entry()
        if e.metadata.mode != "merge":
            continue
        p = t.render_prompt(e.metadata)
        if p in seen:
            assert seen[p] == e.answer
        else:
            seen[p] = e.answer
