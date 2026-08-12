#!/usr/bin/env python3
"""
paired_stats.py

The formal statistical test for the paper's central question: does
LLM-synthetic task data behave like procedurally-generated task data, in
terms of measured training influence?

Design (matches the original experiment plan): treat each TASK as one paired
observation. For a given metric (e.g. BBH) and model scale, compute each
task's mean delta under the procedural arm and its mean delta under the
LLM-synthetic arm (averaging across the seeds already collected for each) --
that gives one paired observation per task (7 tasks = n=7 pairs). Run:

  - Paired t-test: is the mean of (procedural - synthetic) across tasks
    significantly different from zero? (parametric; assumes the paired
    differences are roughly normal, a real stretch at n=7 -- reported
    alongside the non-parametric test below, not instead of it)
  - Wilcoxon signed-rank test: the non-parametric equivalent, safer given
    n=7 is too small to trust a normality assumption
  - Pearson correlation: do the two arms agree on MAGNITUDE, linearly?
  - Spearman correlation: do the two arms agree on which tasks matter most
    (RANKING), independent of exact magnitude?

Repeated per metric, per model scale (n=7 pairs each), and pooled across all
3 model scales per metric (n=21 pairs, more statistical power, at the cost
of mixing scales together).

IMPORTANT interpretation note: for a "these are substitutable" conclusion,
you WANT the paired test's p-value to be LARGE (fail to reject "no
difference between arms") AND the correlation to be high and significant.
A significant paired-test result (p < 0.05, rejecting equality) is evidence
AGAINST substitutability for that metric/scale. A high, significant
correlation is evidence the two arms at least agree on which tasks matter
most, even if their absolute magnitudes differ -- the two numbers can and do
point in different directions for some rows below; report both, don't
collapse to one verdict.

A real, stated limitation, not glossed over: n=7 tasks (or n=21 pooled) is a
small sample for any of these tests -- treat p-values here as indicative,
not as strict significance in the way a much larger-n study would allow.
This is itself worth a sentence in the paper's methods/limitations section.

Input: all_values.csv, produced by build_report.py (columns: task, source,
model, seed, metric, value). Run any time after build_report.py:

    python synthetic/paired_stats.py --csv task_diagnostics/report/all_values.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats as sstats
except ImportError:
    raise SystemExit("This needs scipy: pip install scipy --break-system-packages")

METRICS = ["bbh", "mmlu_math", "mmlu_logic", "mbpp", "dolci", "fw"]
PRIMARY_METRICS = ["bbh", "mmlu_math", "mmlu_logic"]  # the reasoning-transfer metrics


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["value"] = float(r["value"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def task_means(rows, metric, model, source):
    """{task: mean(value)} across whatever seeds exist, for one (metric, model, source)."""
    by_task = defaultdict(list)
    for r in rows:
        if r["metric"] == metric and r["model"] == model and r["source"] == source:
            by_task[r["task"]].append(r["value"])
    return {t: sum(v) / len(v) for t, v in by_task.items()}


def paired_vectors(rows, metric, models):
    """Returns (tasks, proc_vals, synth_vals) for the given metric, pooling
    across the given list of models -- one pair per (task, model) combo
    present in BOTH sources."""
    tasks, proc_vals, synth_vals = [], [], []
    for model in models:
        proc = task_means(rows, metric, model, "procedural")
        synth = task_means(rows, metric, model, "llm_synth")
        for task in sorted(set(proc) & set(synth)):
            tasks.append(f"{task}@{model.split('/')[-1]}")
            proc_vals.append(proc[task])
            synth_vals.append(synth[task])
    return tasks, proc_vals, synth_vals


def run_tests(proc_vals, synth_vals):
    n = len(proc_vals)
    out = dict(n=n)
    if n < 3:
        out["error"] = f"n={n} too small for any of these tests"
        return out
    diffs = [p - s for p, s in zip(proc_vals, synth_vals)]
    try:
        t_res = sstats.ttest_rel(proc_vals, synth_vals)
        out["paired_t_stat"], out["paired_t_p"] = float(t_res.statistic), float(t_res.pvalue)
    except Exception as e:
        out["paired_t_error"] = str(e)
    try:
        if all(d == 0 for d in diffs):
            out["wilcoxon_error"] = "all paired differences are exactly zero"
        else:
            w_res = sstats.wilcoxon(proc_vals, synth_vals)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(w_res.statistic), float(w_res.pvalue)
    except Exception as e:
        out["wilcoxon_error"] = str(e)
    try:
        r, p = sstats.pearsonr(proc_vals, synth_vals)
        out["pearson_r"], out["pearson_p"] = float(r), float(p)
    except Exception as e:
        out["pearson_error"] = str(e)
    try:
        rho, p = sstats.spearmanr(proc_vals, synth_vals)
        out["spearman_rho"], out["spearman_p"] = float(rho), float(p)
    except Exception as e:
        out["spearman_error"] = str(e)
    out["mean_abs_diff"] = sum(abs(d) for d in diffs) / n
    out["mean_diff"] = sum(diffs) / n
    return out


def fmt_p(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def verdict(res):
    if "error" in res:
        return res["error"]
    p_paired = res.get("wilcoxon_p", res.get("paired_t_p"))
    r = res.get("pearson_r")
    bits = []
    if p_paired is not None:
        bits.append("no significant difference (consistent with substitutability)"
                     if p_paired >= 0.05 else
                     "significant difference between arms (evidence AGAINST substitutability)")
    if r is not None:
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        bits.append(f"{strength} correlation (r={r:.2f})")
    return "; ".join(bits) if bits else "insufficient data"


def print_block(title, tasks, proc_vals, synth_vals, res):
    print(f"\n{'=' * 78}\n{title}  (n={res.get('n', 0)})\n{'=' * 78}")
    if "error" in res:
        print(f"  {res['error']}")
        return
    print(f"  Paired t-test:        t={res.get('paired_t_stat', float('nan')):+.3f}  "
          f"p={fmt_p(res.get('paired_t_p'))}")
    if "wilcoxon_p" in res:
        print(f"  Wilcoxon signed-rank: W={res.get('wilcoxon_stat', float('nan')):.3f}  "
              f"p={fmt_p(res.get('wilcoxon_p'))}")
    else:
        print(f"  Wilcoxon signed-rank: {res.get('wilcoxon_error', 'n/a')}")
    print(f"  Pearson correlation:  r={res.get('pearson_r', float('nan')):+.3f}  "
          f"p={fmt_p(res.get('pearson_p'))}")
    print(f"  Spearman correlation: rho={res.get('spearman_rho', float('nan')):+.3f}  "
          f"p={fmt_p(res.get('spearman_p'))}")
    print(f"  Mean |procedural - synthetic|: {res.get('mean_abs_diff', float('nan')):.4f}")
    print(f"  Verdict: {verdict(res)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="task_diagnostics/report/all_values.csv")
    ap.add_argument("--models", nargs="+", default=[
        "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B",
    ])
    ap.add_argument("--out", default="task_diagnostics/report/paired_stats.md")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.csv}")

    md = ["# Paired Statistical Comparison — Procedural vs. LLM-Synthetic\n",
          f"n note: each test below pairs task-level means (averaged across seeds already "
          f"collected per task); per-model tests have n=7 (one per task), pooled-across-model "
          f"tests have n up to 21. Treat these as indicative given the small n, not as strict "
          f"significance in the large-sample sense.\n"]

    print("PRIMARY (reasoning-transfer) METRICS — per model scale")
    for metric in PRIMARY_METRICS:
        md.append(f"\n## {metric.upper()} — per model scale\n")
        md.append("| Model | n | Wilcoxon p | Pearson r (p) | Spearman rho (p) | Mean |diff| | Verdict |")
        md.append("|---|---|---|---|---|---|---|")
        for model in args.models:
            tasks, pv, sv = paired_vectors(rows, metric, [model])
            res = run_tests(pv, sv)
            print_block(f"{metric} @ {model}", tasks, pv, sv, res)
            md.append(f"| {model.split('/')[-1]} | {res.get('n', 0)} | "
                      f"{fmt_p(res.get('wilcoxon_p'))} | "
                      f"{res.get('pearson_r', float('nan')):+.2f} ({fmt_p(res.get('pearson_p'))}) | "
                      f"{res.get('spearman_rho', float('nan')):+.2f} ({fmt_p(res.get('spearman_p'))}) | "
                      f"{res.get('mean_abs_diff', float('nan')):.4f} | {verdict(res)} |")

        tasks, pv, sv = paired_vectors(rows, metric, args.models)
        res = run_tests(pv, sv)
        print_block(f"{metric} — POOLED across all 3 model scales", tasks, pv, sv, res)
        md.append(f"| **pooled (all scales)** | {res.get('n', 0)} | "
                  f"{fmt_p(res.get('wilcoxon_p'))} | "
                  f"{res.get('pearson_r', float('nan')):+.2f} ({fmt_p(res.get('pearson_p'))}) | "
                  f"{res.get('spearman_rho', float('nan')):+.2f} ({fmt_p(res.get('spearman_p'))}) | "
                  f"{res.get('mean_abs_diff', float('nan')):.4f} | **{verdict(res)}** |")

    print("\n\nSECONDARY (mbpp / safety) METRICS — pooled across all model scales only")
    md.append(f"\n## Secondary metrics (mbpp, dolci, fw) — pooled across all model scales\n")
    md.append("| Metric | n | Wilcoxon p | Pearson r (p) | Spearman rho (p) | Mean |diff| | Verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for metric in ["mbpp", "dolci", "fw"]:
        tasks, pv, sv = paired_vectors(rows, metric, args.models)
        res = run_tests(pv, sv)
        print_block(f"{metric} — POOLED across all 3 model scales", tasks, pv, sv, res)
        md.append(f"| {metric} | {res.get('n', 0)} | {fmt_p(res.get('wilcoxon_p'))} | "
                  f"{res.get('pearson_r', float('nan')):+.2f} ({fmt_p(res.get('pearson_p'))}) | "
                  f"{res.get('spearman_rho', float('nan')):+.2f} ({fmt_p(res.get('spearman_p'))}) | "
                  f"{res.get('mean_abs_diff', float('nan')):.4f} | {verdict(res)} |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()