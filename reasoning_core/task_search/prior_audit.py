"""Constant-guess baseline per task/level: how much reward a model gets for free."""
import sys, collections, statistics, reasoning_core as r

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LEVELS = (0, 3, 6)
for name in sys.argv[2:]:
    t = r.get_task(name)
    for L in LEVELS:
        t.config.set_level(L)
        es = [t.generate() for _ in range(N)]
        answers = [str(e.answer) for e in es]
        top, ntop = collections.Counter(answers).most_common(1)[0]
        # constant guess = most frequent answer, scored against every entry
        base = statistics.mean(t.score_answer(top, e) for e in es)
        print(f"{name:28s} L{L}  distinct={len(set(answers))/N:4.2f}"
              f"  majority={ntop/N:4.2f}  const_reward={base:4.2f}"
              f"  len={statistics.mean(map(len, answers)):5.1f}  ex={top[:24]!r}")
