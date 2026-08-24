"""How many items does each eval leg actually need to rank aux tasks the same?

Leg-level pruning (v8 -> v9) cut 3 legs and only 4.8% of the items, because the retired legs were the
small ones. The cost sits in a handful of large legs, so the question is not only WHICH legs to keep
but how many items each needs. That is measurable from the per-example sidecars: subsample a leg,
recompute its per-arm mean, and see when the arm RANKING stops matching the full-size leg.

    python -m reasoning_core.reports.leg_sizing --per-example <dir> --cells '<glob>'

Reports, per leg, the smallest item count whose ranking holds at rho >= --target against the full
leg, median over --reps random draws. A leg that needs 200 of its 1200 items is 83% waste; a leg that
never reaches the target at any size is telling you it has no stable ranking to preserve.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from reasoning_core.reports.item_factors import load_sidecars, join_arms

SIZES = [50, 100, 150, 200, 300, 400, 600, 800, 1200]


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-example", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out")
    a = ap.parse_args()
    import numpy as np

    hits, _ = join_arms(load_sidecars(a.per_example), a.cells)
    arms = [(lbl, s) for lbl, s in hits if lbl[0] != "__BASELINE__"]
    if len(arms) < 6:
        raise SystemExit(f"[sizing] only {len(arms)} arms joined; need >= 6 to rank")
    legs = sorted(set.intersection(*[set(s["legs"]) for _, s in arms]))
    rng = np.random.default_rng(43)
    plan, total_full, total_cut = {}, 0, 0
    print(f"[sizing] {len(arms)} arms, target rho {a.target}, {a.reps} draws per size\n")
    print(f"{'leg':<20}{'items':>7}{'needed':>8}{'keep':>7}  rho@need")
    for leg in legs:
        field = "margin" if "margin" in arms[0][1]["legs"][leg] else "nll"
        M = np.array([[v if v is not None else np.nan
                       for v in s["legs"][leg][field]] for _, s in arms], dtype=float)
        M = M[:, ~np.isnan(M).any(axis=0)]
        n = M.shape[1]
        if n < 40:
            plan[leg] = n; total_full += n; total_cut += n
            print(f"{leg:<20}{n:>7}{n:>8}{'100%':>7}  (too small to subsample)")
            continue
        full = M.mean(axis=1).tolist()
        need, got = n, 1.0
        for sz in [s for s in SIZES if s < n]:
            rhos = [spearman(M[:, rng.choice(n, sz, replace=False)].mean(axis=1).tolist(), full)
                    for _ in range(a.reps)]
            med = float(np.median(rhos))
            if med >= a.target:
                need, got = sz, med
                break
        plan[leg] = need
        total_full += n
        total_cut += need
        print(f"{leg:<20}{n:>7}{need:>8}{100*need/n:>6.0f}%  {got:+.3f}")
    print(f"\n[sizing] {total_full} -> {total_cut} items  ({100*(1-total_cut/total_full):.1f}% cut)")
    if a.out:
        Path(a.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"[sizing] plan -> {a.out}")


if __name__ == "__main__":
    main()
