#!/usr/bin/env python3
"""
build_report.py

Reads every per_task_results/influence_*.json produced by
run_experiment_grid.py, aggregates across seeds (mean +/- std) per
(task, model, source, metric), and writes a publication-ready report:

  - Per-model-size tables: procedural vs LLM-synthetic, side by side, for
    every task and every metric (bbh/mmlu_math/mmlu_logic/mbpp/dolci/fw).
  - A scaling table: how each task's influence changes across model sizes,
    per source.
  - An agreement summary: for tasks with BOTH sources, how close are they
    (mean absolute difference, sign agreement) -- the core "does synthetic
    substitute for procedural" evidence.
  - A CSV of every (task, model, source, seed, metric) value, for your own
    further analysis/plotting.

Run this ON THE GPU MACHINE, any time, as often as you like (it only reads
existing result files, never launches anything):

    python synthetic/build_report.py
    python synthetic/build_report.py --out-dir report_for_paper
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = [
    "code_execution", "code_runnability", "code_analysis", "code_input_deduction",
    "temporal_reasoning", "type_inhabitation", "code_repair",
]
METRICS = ["bbh", "mmlu_math", "mmlu_logic", "mbpp", "dolci", "fw"]

# Matches how run_experiment_grid.py builds run_tag: "{SOURCE}_{model_tag}"
RUN_TAG_RE = re.compile(r"^(PROCEDURAL|LLM_SYNTH)_(.+)$")


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        return None


def parse_source_model(data, filename):
    """Recover (source, model_id) for one result file: prefer the file's own
    recorded fields, fall back to parsing it out of the filename's run_tag
    if needed."""
    model = data.get("model")
    m = re.match(r"influence_(.+?)_S\d+_T\d+_M\d+_", filename)
    run_tag = m.group(1) if m else filename
    tm = RUN_TAG_RE.match(run_tag)
    source = {"PROCEDURAL": "procedural", "LLM_SYNTH": "llm_synth"}.get(tm.group(1)) if tm else None
    if model is None and tm:
        model = tm.group(2)  # short tag, e.g. "SmolLM2-135M" -- best effort
    return source, model


def collect(results_dir):
    """Returns rows: list of dicts, one per (task, source, model, seed, metric)."""
    rows = []
    for path in sorted(Path(results_dir).glob("influence_*.json")):
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), dict):
            continue
        source, model = parse_source_model(data, path.name)
        seed = data.get("seed")
        if source is None or model is None or seed is None:
            print(f"warning: couldn't parse source/model/seed from {path.name}, skipping", file=sys.stderr)
            continue
        for task_name, rec in data["tasks"].items():
            if task_name not in TASKS or not isinstance(rec, dict):
                continue
            for metric in METRICS:
                key = f"{metric}_delta"
                if key in rec and isinstance(rec[key], (int, float)):
                    rows.append(dict(task=task_name, source=source, model=model, seed=seed,
                                      metric=metric, value=float(rec[key]), file=path.name))
            for extra_key in ("acc0", "acc_final", "solve0", "solve_final"):
                if extra_key in rec and isinstance(rec[extra_key], (int, float)):
                    rows.append(dict(task=task_name, source=source, model=model, seed=seed,
                                      metric=extra_key, value=float(rec[extra_key]), file=path.name))
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    return stats.stdev(xs) if len(xs) > 1 else 0.0


def aggregate(rows):
    """(task, source, model, metric) -> {values, mean, std, n}"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["source"], r["model"], r["metric"])].append(r["value"])
    out = {}
    for key, values in groups.items():
        out[key] = dict(values=values, mean=mean(values), std=sd(values), n=len(values))
    return out


def model_order(models):
    """Sort by parameter count parsed out of the model name (135M < 360M < 1.7B),
    falling back to alphabetical if it can't parse a size."""
    def size_of(m):
        mm = re.search(r"(\d+(?:\.\d+)?)([MB])", m)
        if not mm:
            return float("inf")
        val, unit = float(mm.group(1)), mm.group(2)
        return val * (1000 if unit == "B" else 1)
    return sorted(models, key=size_of)


def fmt(x, digits=4, signed=True):
    if x is None or not isinstance(x, (int, float)) or x != x:  # NaN check
        return "--"
    sign = "+" if signed else ""
    return f"{x:{sign}.{digits}f}"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "source", "model", "seed", "metric", "value", "file"])
        w.writeheader()
        w.writerows(rows)


def build_per_model_tables(agg, models, sources, tasks):
    """One table per model size: task x metric, procedural vs llm_synth side by side."""
    lines = []
    for model in models:
        lines.append(f"### {model}\n")
        header = ["task"]
        for metric in METRICS:
            header += [f"{metric} (proc)", f"{metric} (synth)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for task in tasks:
            row = [task]
            for metric in METRICS:
                cells = []
                for source in ("procedural", "llm_synth"):
                    a = agg.get((task, source, model, metric))
                    if a is None:
                        cells.append("--")
                    elif a["n"] > 1:
                        cells.append(f"{fmt(a['mean'])} ± {fmt(a['std'], 4, False)} (n={a['n']})")
                    else:
                        cells.append(f"{fmt(a['mean'])} (n=1)")
                row += cells
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def build_scaling_table(agg, models, sources, tasks, metric):
    """One row per task, one column per model size, for a single metric -- the
    core 'does this task's influence hold up as the model scales' evidence."""
    lines = [f"### Scaling: {metric}_delta across model size\n"]
    header = ["task", "source"] + models
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for task in tasks:
        for source in sources:
            row = [task, source]
            for model in models:
                a = agg.get((task, source, model, metric))
                row.append(fmt(a["mean"]) if a else "--")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def build_agreement_table(agg, models, tasks):
    """For tasks present on BOTH sources: mean abs difference + sign agreement,
    per model size, per metric -- the direct 'substitutability' evidence."""
    lines = ["### Procedural vs LLM-synthetic agreement (tasks with both sources)\n"]
    header = ["task", "model", "metric", "procedural", "llm_synth", "abs diff", "same sign?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    any_row = False
    for task in tasks:
        for model in models:
            for metric in METRICS:
                p = agg.get((task, "procedural", model, metric))
                s = agg.get((task, "llm_synth", model, metric))
                if p is None or s is None:
                    continue
                any_row = True
                diff = abs(p["mean"] - s["mean"])
                same_sign = "yes" if (p["mean"] * s["mean"]) >= 0 else "no"
                lines.append("| " + " | ".join([
                    task, model, metric, fmt(p["mean"]), fmt(s["mean"]), fmt(diff, 4, False), same_sign,
                ]) + " |")
    if not any_row:
        lines.append("| _(no task currently has both sources for the same model -- run both to populate this)_ | | | | | | |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "per_task_results"))
    ap.add_argument("--out-dir", default=str(ROOT / "task_diagnostics" / "report"))
    ap.add_argument("--primary-metric", default="bbh",
                     help="Metric used for the scaling table headline (others still get their own table).")
    args = ap.parse_args()

    rows = collect(args.results_dir)
    if not rows:
        raise SystemExit(f"No usable influence_*.json rows found under {args.results_dir} -- "
                          f"has run_experiment_grid.py finished any runs yet?")

    agg = aggregate(rows)
    models_present = model_order(sorted({r["model"] for r in rows}))
    sources_present = sorted({r["source"] for r in rows})
    tasks_present = [t for t in TASKS if any(r["task"] == t for r in rows)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_dir / "all_values.csv")

    md = []
    md.append("# Procedural vs LLM-Synthetic Task Influence -- Results\n")
    md.append(f"Models found: {', '.join(models_present)}\n")
    md.append(f"Sources found: {', '.join(sources_present)}\n")
    md.append(f"Tasks found: {', '.join(tasks_present)}\n")
    md.append(f"Total (task, source, model, seed, metric) rows aggregated: {len(rows)}\n")
    md.append("\nLower delta = reduced held-out loss = the task helped (same convention as "
              "task_influence.py's own reports).\n")

    md.append("\n## Per-model-size breakdown\n")
    md.append(build_per_model_tables(agg, models_present, sources_present, tasks_present))

    md.append("\n## Scaling across model size\n")
    for metric in METRICS:
        md.append(build_scaling_table(agg, models_present, sources_present, tasks_present, metric))

    md.append("\n## Procedural vs LLM-synthetic agreement\n")
    md.append(build_agreement_table(agg, models_present, tasks_present))

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(md))

    # Machine-readable sidecar too, for anyone plotting this in their own tooling.
    agg_json = {
        f"{task}|{source}|{model}|{metric}": v
        for (task, source, model, metric), v in agg.items()
    }
    (out_dir / "aggregated.json").write_text(json.dumps(agg_json, indent=2, sort_keys=True))

    print(f"{len(rows)} rows from {len(set(r['file'] for r in rows))} result file(s)")
    print(f"Models: {models_present}")
    print(f"Sources: {sources_present}")
    print(f"Tasks: {tasks_present}")
    print(f"\nWrote:")
    print(f"  {report_path}")
    print(f"  {out_dir / 'aggregated.json'}")
    print(f"  {out_dir / 'all_values.csv'}")


if __name__ == "__main__":
    main()