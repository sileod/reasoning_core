"""Which edited tasks actually CHANGED BEHAVIOUR, versus were merely edited.

`behavior_hash` is the sha1 of a task module's docstring-stripped AST, so it flags any code edit --
after one hardening commit, 50 of 51 pool tasks were "changed". Re-measuring all of them is a full
roster fleet; most of those edits are probably cosmetic.

Byte-comparing regenerated rows does NOT work: generation is not reproducible from random.seed(), so
identical code yields different rows. Compare DISTRIBUTIONS instead. If an edit was cosmetic, the
new rows should be statistically indistinguishable from the pool rows on the features that drive
influence -- prompt and answer length (length dominates influence rankings) and the answer character
class (answer format drives the held-out LM tax).

    python -m reasoning_core.reports.roster_drift --pool 4d670fac3956 --new roster_v11/<id>

Prints the tasks that moved, which is the set that actually needs re-measuring.
"""
from __future__ import annotations
import argparse, collections, glob, math


def rows(cache):
    import pyarrow.parquet as pq
    out = collections.defaultdict(lambda: {"plen": [], "alen": [], "charset": collections.Counter()})
    for f in glob.glob(f"{cache}/data/*.parquet"):
        d = pq.read_table(f, columns=["task", "prompt", "answer"]).to_pydict()
        for t, p, a in zip(d["task"], d["prompt"], d["answer"]):
            r = out[t]
            r["plen"].append(len(str(p)))
            r["alen"].append(len(str(a)))
            r["charset"][_cls(str(a))] += 1
    return out


def _cls(a):
    """Coarse answer-format class -- the axis with a measured transfer effect."""
    a = a.strip()
    if not a:
        return "empty"
    if len(a) == 1:
        return "single_char"
    if a.replace("-", "").replace(".", "").isdigit():
        return "number"
    if a.isalpha():
        return "word"
    if any(c in a for c in "()[]{}|=<>+*/^"):
        return "symbolic"
    return "mixed"


def ks(x, y):
    """Two-sample Kolmogorov-Smirnov statistic, no scipy."""
    if not x or not y:
        return 1.0
    xs, ys = sorted(x), sorted(y)
    allv = sorted(set(xs) | set(ys))
    import bisect
    d = 0.0
    for v in allv:
        fx = bisect.bisect_right(xs, v) / len(xs)
        fy = bisect.bisect_right(ys, v) / len(ys)
        d = max(d, abs(fx - fy))
    return d


def ks_crit(n, m, alpha=0.01):
    return 1.63 * math.sqrt((n + m) / (n * m)) if n and m else 1.0


def tvd(a, b):
    """Total variation distance between two answer-class distributions."""
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--tvd-max", type=float, default=0.05)
    a = ap.parse_args()
    old, new = rows(a.pool), rows(a.new)
    moved, same, added = [], [], []
    print(f"{'task':<32}{'KS plen':>9}{'crit':>7}{'KS alen':>9}{'fmt TVD':>9}  verdict")
    for t in sorted(set(old) | set(new)):
        if t not in old:
            added.append(t); continue
        if t not in new:
            continue
        o, n = old[t], new[t]
        kp, ka = ks(o["plen"], n["plen"]), ks(o["alen"], n["alen"])
        crit = ks_crit(len(o["plen"]), len(n["plen"]))
        fmt = tvd(o["charset"], n["charset"])
        drift = kp > crit or ka > crit or fmt > a.tvd_max
        (moved if drift else same).append(t)
        print(f"{t:<32}{kp:>9.3f}{crit:>7.3f}{ka:>9.3f}{fmt:>9.3f}  {'MOVED' if drift else 'same'}")
    print(f"\nmoved {len(moved)} · indistinguishable {len(same)} · new {len(added)}")
    print(f"\nRE-MEASURE ({len(moved)+len(added)}): {' '.join(sorted(moved + added))}")
    print(f"\nKEEP EXISTING CELLS ({len(same)}): {' '.join(sorted(same))}")


if __name__ == "__main__":
    main()
