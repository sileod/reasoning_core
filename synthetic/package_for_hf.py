#!/usr/bin/env python3
"""
package_for_hf.py

Takes everything accumulated in generated_data/llm_tasks/*.jsonl (written
incrementally, across as many llm_generate_tasks.py runs as you like) and
turns it into a clean, checked, HuggingFace-ready dataset directory.

What "checked" means here, concretely:
  1. Re-validates every row by reconstructing it as a real reasoning_core
     Entry and calling that task's own, real score_answer(answer, entry) --
     i.e. re-running the actual grader against its own reference answer.
     A row only survives this if it still scores a perfect 1.0. This catches
     any JSON round-tripping issues (e.g. tuples silently becoming lists)
     or pipeline bugs between generation time and packaging time -- it does
     NOT re-verify "is this good code" (that already happened, for every
     row, at generation time, inside the task's own sandboxed pipeline
     before the row was ever written).
  2. Deduplicates across every run you've done so far (by task+prompt+answer),
     since the JSONL files accumulate across multiple invocations.
  3. Writes one clean file per task plus a combined file, and an auto-generated
     dataset card (README.md) with row counts, level distribution, and source
     model, so a `datasets.load_dataset(...)` / `push_to_hub(...)` on it is
     unsurprising to whoever reads it later.

Run:
    python scripts/package_for_hf.py
    python scripts/package_for_hf.py --push-to-hub your-username/reasoning-core-llm-gen
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasoning_core import get_task  # noqa: E402
from reasoning_core.template import Entry, edict  # noqa: E402
import reasoning_core as _rc
print(f"(using reasoning_core from: {_rc.__file__})")

import re  # noqa: E402

REAL_TASK_NAME = {"code_iterations": "temporal_reasoning"}
TOOLKIT_TASKS = {"type_inhabitation", "code_repair"}


def _parse_toolkit_text(toolkit_text):
    """Reverses TypeInhabitation/CodeRepair's own rendering of toolkit_text
    ("  f3(pred: Prediction, truth: Blob) -> float") back into
    {masked_name: (inputs, output)}."""
    records = {}
    for line in toolkit_text.splitlines():
        m = re.match(r"\s*(\w+)\((.*)\)\s*->\s*(.+)$", line)
        if not m:
            continue
        name, inputs_str, output = m.group(1), m.group(2).strip(), m.group(3).strip()
        inputs = []
        for part in filter(None, (p.strip() for p in inputs_str.split(","))):
            pname, _, ptype = part.partition(":")
            inputs.append((pname.strip(), ptype.strip()))
        records[name] = (inputs, output)
    return records


def _parse_vars_text(vars_text):
    var_types = {}
    for line in vars_text.splitlines():
        if ":" not in line:
            continue
        name, _, typ = line.strip().partition(":")
        var_types[name.strip()] = typ.strip()
    return var_types


def reconstruct_toolkit_metadata(metadata):
    """func_lookup (a dict of FunctionRecord objects) doesn't survive JSON
    round-tripping -- task_diagnostics.schemas.canonical_json falls back to
    str(obj) for anything it can't natively serialize (see its `default=str`),
    which is a property of the existing task_diagnostics cache format for
    these two tasks, not something specific to LLM-generated rows. Rebuild it
    from toolkit_text/vars_text/reverse_map instead, which ARE plain,
    JSON-safe strings/dicts and preserve everything needed."""
    from reasoning_core.tasks.code_reasoning import FunctionRecord

    masked_records = _parse_toolkit_text(metadata["toolkit_text"])
    reverse_map = metadata["reverse_map"]  # masked_name -> original_name, plain str->str
    func_lookup = {}
    for masked_name, (inputs, output) in masked_records.items():
        original_name = reverse_map.get(masked_name, masked_name)
        func_lookup[original_name] = FunctionRecord(original_name, inputs, output)
    var_types = _parse_vars_text(metadata["vars_text"])
    return func_lookup, var_types


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedup_key(row):
    raw = f"{row['task']}|{row['prompt']}|{row['answer']}"
    return hashlib.sha1(raw.encode()).hexdigest()


def revalidate(task, row, public_name):
    """Rebuild a real Entry from the saved row and re-run the task's own
    score_answer against its own reference answer. Returns (ok, score)."""
    try:
        meta = dict(row["metadata"])
        if public_name in TOOLKIT_TASKS:
            func_lookup, var_types = reconstruct_toolkit_metadata(meta)
            meta["func_lookup"] = func_lookup
            meta["var_types"] = var_types
        entry = Entry(metadata=edict(meta), answer=row["answer"])
        score = task.score_answer(row["answer"], entry)
        return (float(score) == 1.0), score
    except Exception as e:
        return False, f"error: {e}"


def make_dataset_card(stats, model_names, out_dir):
    lines = [
        "# LLM-generated reasoning-core tasks",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')} using "
        f"{', '.join(sorted(model_names)) or 'an LLM'} via `litlm`, following the same "
        "task pipelines (sandboxed execution, type-checking, acceptance criteria) as the "
        "procedural (mesopy) version of reasoning-core, for a direct, apples-to-apples "
        "comparison against it.",
        "",
        "Every row passed the task's own real validation pipeline at generation time "
        "(not just a superficial format check), and was re-verified again at packaging "
        "time by re-running the task's own scorer against its own reference answer.",
        "",
        "## Rows per task",
        "",
        "| task | rows | levels |",
        "|---|---|---|",
    ]
    for task, info in sorted(stats.items()):
        levels = ", ".join(str(l) for l in sorted(info["levels"]))
        lines.append(f"| {task} | {info['n']} | {levels} |")
    lines += [
        "",
        "## Fields",
        "",
        "- `task`: task name",
        "- `level`: difficulty level (0-4)",
        "- `prompt`: the rendered prompt",
        "- `answer`: the reference answer",
        "- `metadata`: task-specific structured fields backing the prompt/answer",
        "- `call_id` / `source`: provenance back to the exact LLM generation call "
        "(`null`/`\"synthetic-pool\"` for type_inhabitation/code_repair rows sampled "
        "from the shared synthetic function pool rather than one LLM call per row)",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl-root", default="generated_data/llm_tasks")
    ap.add_argument("--out-dir", default="generated_data/hf_dataset")
    ap.add_argument("--push-to-hub", default=None,
                     help="If given (e.g. 'your-username/dataset-name'), also push via "
                          "huggingface_hub after packaging. Requires `pip install huggingface_hub` "
                          "and `huggingface-cli login` first.")
    args = ap.parse_args()

    jsonl_root = Path(args.jsonl_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(jsonl_root.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found under {jsonl_root} -- run llm_generate_tasks.py first.")

    stats = {}
    model_names = set()
    combined = []

    for path in files:
        public_name = path.stem
        real_name = REAL_TASK_NAME.get(public_name, public_name)
        task = get_task(real_name)

        rows = load_jsonl(path)
        seen_keys = set()
        kept, dropped_dupe, dropped_invalid = [], 0, 0

        for row in rows:
            key = dedup_key(row)
            if key in seen_keys:
                dropped_dupe += 1
                continue
            seen_keys.add(key)

            ok, score = revalidate(task, row, public_name)
            if not ok:
                dropped_invalid += 1
                print(f"  [{public_name}] DROPPED a row that failed re-validation "
                      f"(score={score!r}): {row['prompt'][:80]!r}...")
                continue

            kept.append(row)
            if row.get("source"):
                model_names.add(str(row["source"]))

        out_path = out_dir / f"{public_name}.jsonl"
        with open(out_path, "w") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")

        levels = Counter(r["level"] for r in kept)
        stats[public_name] = dict(n=len(kept), levels=set(levels))
        combined.extend(kept)

        print(f"{public_name:24s} kept={len(kept):5d}  dropped_dupe={dropped_dupe:4d}  "
              f"dropped_failed_revalidation={dropped_invalid:4d}  levels={dict(sorted(levels.items()))}")

    combined_path = out_dir / "all_tasks.jsonl"
    with open(combined_path, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    make_dataset_card(stats, model_names, out_dir)

    print(f"\n{len(combined)} total rows across {len(files)} tasks -> {out_dir}")
    print(f"  - per-task files: {out_dir}/<task>.jsonl")
    print(f"  - combined file:  {combined_path}")
    print(f"  - dataset card:   {out_dir}/README.md")
    print(f"\nSanity-load it yourself with:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('json', data_files='{combined_path}')")

    if args.push_to_hub:
        push(args.push_to_hub, out_dir)


def push(repo_id, out_dir):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("--push-to-hub needs the `datasets` package: pip install datasets huggingface_hub")
    print(f"\nLoading {out_dir}/all_tasks.jsonl and pushing to {repo_id} ...")
    ds = load_dataset("json", data_files=str(out_dir / "all_tasks.jsonl"))
    ds.push_to_hub(repo_id)
    print(f"Pushed. Also upload {out_dir}/README.md as the dataset card if it wasn't picked up automatically:")
    print(f"  from huggingface_hub import upload_file")
    print(f"  upload_file(path_or_fileobj='{out_dir}/README.md', path_in_repo='README.md', "
          f"repo_id='{repo_id}', repo_type='dataset')")


if __name__ == "__main__":
    main()