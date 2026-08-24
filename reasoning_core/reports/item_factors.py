"""Exploratory factor analysis over ITEMS, not legs: which eval questions move together?

The leg-level varimax in the Atlas answers "which benchmarks are redundant". This answers the finer
question the leg means cannot: within a leg, do all items carry the same signal, or is a leg a
mixture of item clusters that respond to different aux tasks? That is the prerequisite for building a
cheap battery by SELECTING items rather than dropping whole legs.

Input is the per-example sidecars written by `--per-example-dir` (item-level margin / gold_nll /
nll per leg). Arms are joined to tasks through the metric vector stored in each sidecar, since the
filename is only a content hash.

    python -m reasoning_core.reports.item_factors --per-example <dir> --cells 'influence_COLL-roster_WLSH*'

Method: build an items x arms matrix of per-item deltas against the shared baseline arm, z-score per
item, extract components, then varimax-rotate. Unrotated components are dominated by "this item is
easy" and are not interpretable.

`--k` is fixed at 4 by DEFAULT because that is the resolution the current arm count supports and a
default that silently changes with the data makes two runs incomparable. Pass `--select` to choose
`k` by parallel analysis instead: the same matrix is re-run with each item row independently
shuffled across arms, which destroys cross-arm structure while preserving each item's own
distribution, and components are kept only while they beat the 95th percentile of that null. Raising
`--k` past the retained count is allowed and useful for splitting a broad factor, but components
below the null line are noise and are reported as such.
"""
from __future__ import annotations
import argparse, glob, gzip, json, math
from pathlib import Path

PR = Path(__file__).resolve().parents[2] / "per_task_results"


def load_sidecars(d):
    out = []
    for f in sorted(glob.glob(str(Path(d) / "*.json.gz"))):
        try:
            out.append((f, json.loads(gzip.decompress(Path(f).read_bytes()))))
        except Exception:
            continue
    return out


def join_arms(sidecars, cell_glob):
    """Map each sidecar to (task, seed) by matching its stored metric vector against the cells.

    The cell stores the metric vector PLUS the derived `_delta` aliases (189 keys vs the sidecar's
    89), so the two key sets never match outright. Project the cell down onto whatever keys the
    sidecar actually carries and compare on that.
    """
    cells = []
    for f in glob.glob(str(PR / cell_glob)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        seed = d.get("seed")
        for task, m in (d.get("tasks") or {}).items():
            cells.append((task, seed, m))
        if d.get("baseline"):
            cells.append(("__BASELINE__", seed, d["baseline"]))

    def key(metrics, names):
        return tuple(round(metrics[n], 9) if isinstance(metrics.get(n), (int, float)) else None
                     for n in names)

    index, hits, misses = {}, [], 0
    for f, s in sidecars:
        names = tuple(sorted(k for k, v in s["metrics"].items() if isinstance(v, (int, float))))
        if names not in index:
            index[names] = {}
            for task, seed, m in cells:
                if all(n in m for n in names):
                    index[names].setdefault(key(m, names), (task, seed))
        got = index[names].get(key(s["metrics"], names))
        if got:
            hits.append((got, s))
        else:
            misses += 1
    return hits, misses


# NLL legs improve when the delta is NEGATIVE, margin legs when POSITIVE. Mixing both in one matrix
# without aligning them makes a factor's arm loadings uninterpretable: a positive loading would mean
# "helps the margin legs and hurts the NLL legs". Flip the NLL rows so up = better everywhere.
def _is_nll(leg_payload):
    return "nll" in leg_payload and "margin" not in leg_payload


def item_matrix(hits, value="margin"):
    """items x arms matrix of per-item values; returns (rows, labels, item_ids)."""
    arms = [(lbl, s) for lbl, s in hits if lbl[0] != "__BASELINE__"]
    base = {s["seed"]: s for lbl, s in hits if lbl[0] == "__BASELINE__"}
    if not arms:
        raise SystemExit("[efa] no treatment arms joined")
    legs = sorted(set.intersection(*[set(s["legs"]) for _, s in arms]))
    ids, cols = [], []
    for name in legs:
        arr0 = arms[0][1]["legs"][name]
        field = value if value in arr0 else ("nll" if "nll" in arr0 else None)
        if field is None:
            continue
        ids += [(name, i) for i in range(len(arr0[field]))]
    for lbl, s in arms:
        b = base.get(s["seed"])
        col = []
        for name, i in ids:
            leg = s["legs"][name]
            field = value if value in leg else "nll"
            v = leg[field][i]
            bv = b["legs"][name][field][i] if b and name in b["legs"] else None
            col.append(None if v is None or bv is None else v - bv)
        cols.append(col)
    keep = [k for k in range(len(ids)) if all(c[k] is not None for c in cols)]
    ref = arms[0][1]["legs"]
    rows = []
    for k in keep:
        row = [cols[j][k] for j in range(len(cols))]
        rows.append([-v for v in row] if _is_nll(ref[ids[k][0]]) else row)
    return rows, [lbl for lbl, _ in arms], [ids[k] for k in keep]


def zscore(rows):
    out = []
    for r in rows:
        n = len(r); m = sum(r) / n
        sd = (sum((x - m) ** 2 for x in r) / n) ** 0.5
        out.append([0.0] * n if sd < 1e-12 else [(x - m) / sd for x in r])
    return out


def pca(rows, k):
    """Top-k components of the items x arms matrix (arms are the variables).

    numpy SVD, not power iteration: parallel analysis re-runs this once per null replicate, and the
    pure-Python version turned a k=8 scan into a ten-minute job.
    """
    import numpy as np
    A = np.asarray(rows, dtype=np.float64)
    k = min(k, min(A.shape))
    # economy SVD of the item x arm matrix; right singular vectors are the arm-space components
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    out = []
    for i in range(k):
        v = vt[i]
        out.append((v.tolist(), float(s[i] ** 2), (A @ v).tolist()))
    return out


def parallel_analysis(Z, k, reps=5, seed=43):
    """How many components beat chance. Shuffle each ITEM row across arms independently: that kills
    every cross-arm correlation while leaving each item's own spread untouched, so the null has the
    same marginal structure as the data and differs only in the thing being tested."""
    import numpy as np
    rng = np.random.default_rng(seed)
    real = [c[1] for c in pca(Z, k)]
    A = np.asarray(Z, dtype=np.float64)
    total = float((A ** 2).sum()) or 1.0
    draws = [[] for _ in range(k)]
    for _ in range(reps):
        S = rng.permuted(A, axis=1)
        for i, c in enumerate(pca(S, k)):
            draws[i].append(c[1] / total)
    null95 = [sorted(d)[min(len(d) - 1, int(0.95 * len(d)))] for d in draws]
    keep = 0
    for i in range(k):
        if real[i] / total > null95[i]:
            keep += 1
        else:
            break
    return keep, null95


def varimax(loadings, iters=60):
    """loadings: list of component vectors over arms. Rotate to simple structure."""
    k = len(loadings); n = len(loadings[0])
    L = [[loadings[c][j] for c in range(k)] for j in range(n)]
    for _ in range(iters):
        for a in range(k):
            for b in range(a + 1, k):
                x = [L[j][a] for j in range(n)]; y = [L[j][b] for j in range(n)]
                u = [xi * xi - yi * yi for xi, yi in zip(x, y)]
                v = [2 * xi * yi for xi, yi in zip(x, y)]
                num = 2 * (n * sum(ui * vi for ui, vi in zip(u, v)) - sum(u) * sum(v))
                den = n * (sum(ui * ui for ui in u) - sum(vi * vi for vi in v)) - (sum(u) ** 2 - sum(v) ** 2)
                if abs(den) < 1e-12 and abs(num) < 1e-12:
                    continue
                phi = 0.25 * math.atan2(num, den)
                c, s = math.cos(phi), math.sin(phi)
                for j in range(n):
                    xa, xb = L[j][a], L[j][b]
                    L[j][a] = c * xa + s * xb
                    L[j][b] = -s * xa + c * xb
    return [[L[j][c] for j in range(n)] for c in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-example", required=True)
    ap.add_argument("--cells", required=True, help="glob under per_task_results for the matching cells")
    ap.add_argument("--k", type=int, default=4,
                    help="number of factors to extract (default 4). Raise it to resolve finer "
                         "structure; components below the parallel-analysis null are flagged.")
    ap.add_argument("--select", action="store_true",
                    help="run parallel analysis against a row-shuffled null and report how many "
                         "components are supported; does not change --k unless --auto is given")
    ap.add_argument("--auto", action="store_true",
                    help="with --select, set k to the number of supported components")
    ap.add_argument("--null-reps", type=int, default=5)
    ap.add_argument("--value", default="margin", choices=("margin", "gold_nll"))
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    side = load_sidecars(a.per_example)
    hits, misses = join_arms(side, a.cells)
    print(f"[efa] {len(side)} sidecars, {len(hits)} joined to arms, {misses} unmatched")
    rows, labels, ids = item_matrix(hits, a.value)
    print(f"[efa] item matrix {len(rows)} items x {len(labels)} arms  (value={a.value})")
    Z = zscore(rows)
    if a.select:
        keep, null95 = parallel_analysis(Z, a.k, a.null_reps)
        print(f"[efa] parallel analysis ({a.null_reps} reps): {keep} of {a.k} components beat the "
              f"null 95th percentile ({', '.join(f'{100*x:.1f}%' for x in null95)})")
        if a.auto and keep:
            a.k = keep
            print(f"[efa] --auto: k set to {keep}")
    comps = pca(Z, a.k)
    total = sum(sum(z * z for z in r) for r in Z) or 1.0
    load = varimax([c[0] for c in comps])
    print(f"[efa] variance explained (unrotated): "
          + ", ".join(f"PC{i+1} {100*c[1]/total:.1f}%" for i, c in enumerate(comps)))
    for c in range(a.k):
        w = sorted(range(len(labels)), key=lambda j: -abs(load[c][j]))[:a.top]
        print(f"\n  factor {c+1}: " + ", ".join(f"{labels[j][0]}({load[c][j]:+.2f})" for j in w))
        pass

    # Which legs EXPRESS each factor. Ranking legs by absolute projection confounds "expresses this
    # factor" with "has large item-level effects at all" -- under that ranking drop and triviaqa came
    # out top-6 on two different factors at once. Report each leg's SHARE of its own explained
    # variance instead, and call a leg factor-dominant only above 45%.
    scores = [[sum(Z[i][j] * load[c][j] for j in range(len(labels))) for c in range(a.k)]
              for i in range(len(Z))]
    agg, cnt = {}, {}
    for (leg, _), s in zip(ids, scores):
        agg.setdefault(leg, [0.0] * a.k)
        for c in range(a.k):
            agg[leg][c] += s[c] * s[c]
        cnt[leg] = cnt.get(leg, 0) + 1
    print(f"\n{'leg':<20}{'n':>6}  " + " ".join(f"F{c+1:<5}" for c in range(a.k)) + " dominant")
    for leg in sorted(agg, key=lambda L: -max(agg[L]) / (sum(agg[L]) or 1)):
        v = agg[leg]
        tot = sum(v) or 1.0
        share = [x / tot for x in v]
        d = max(range(a.k), key=lambda c: share[c])
        mark = "*" if share[d] > 0.45 else ""
        print(f"{leg:<20}{cnt[leg]:>6}  "
              + " ".join(f"{100*share[c]:4.0f}% " for c in range(a.k))
              + f" F{d+1}{mark}")
    dom = {c: [L for L in agg if max(range(a.k), key=lambda x: agg[L][x] / (sum(agg[L]) or 1)) == c
               and agg[L][c] / (sum(agg[L]) or 1) > 0.45] for c in range(a.k)}
    print("\nfactor-dominant legs (>45% of the leg's explained variance):")
    for c in range(a.k):
        print(f"  F{c+1}: " + (", ".join(sorted(dom[c])) if dom[c] else
                               "NONE -- this factor has no distinctive membership, treat as residual"))


if __name__ == "__main__":
    main()
