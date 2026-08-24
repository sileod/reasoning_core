#!/usr/bin/env python3
"""Delete one task's rows from a build so the fleet regenerates just that task.

When a generator is reworked mid-release, its existing rows are wrong but every other task's are
fine. Two properties of the pipeline make a surgical replacement safe, and neither is obvious:

  * `generation_worker` skips a batch only when `<task>-<idx>.jsonl` already exists, so removing one
    task's files makes the workers refill exactly those slots and touch nothing else.
  * `collectv.file_key` is sha1(path + size + mtime_ns), so a regenerated file keys DIFFERENTLY from
    the one it replaced. Collect therefore picks it up on its own -- the state needs no surgery, and
    the stale entries are harmless orphans.

What this does NOT fix: rows already pushed to the dataset. Staging then holds both generations of
the task, separable only by `_task_behavior_hash`, while `cache from-hf` selects by task and level.
Carve with an explicit hash, or rebuild the dataset, before trusting a mixed repo.

    python -m reasoning_core.reports.task_purge --root <gen_root> --tasks code_execution --prefix 'rc11.2-*'
    python -m reasoning_core.reports.task_purge --root <gen_root> --tasks code_execution --apply

Dry-run by default: it prints what it would remove and the hashes those rows carry, so you can
confirm you are deleting the generation you think you are.
"""
from __future__ import annotations
import argparse, collections, glob, json, os


def shard_files(root, prefix, tasks):
    """task -> [paths]. Files are named '<task>-<idx>.jsonl', so the task name is a prefix match up
    to the LAST hyphen -- 'code_execution-12.jsonl' must not be claimed by task 'code'."""
    want = set(tasks)
    out = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, prefix))):
        with os.scandir(d) as it:
            for f in it:
                if not (f.name.endswith(".jsonl") or f.name.endswith(".lock")):
                    continue
                stem = f.name.rsplit(".", 1)[0]
                task = stem.rsplit("-", 1)[0]
                if task in want:
                    out[task].append(f.path)
    return out


def row_hash(path):
    """The generator hash stamped on a file's first row, or None if unreadable."""
    try:
        with open(path) as fh:
            line = fh.readline()
        md = json.loads(line).get("metadata") or {}
        if isinstance(md, str):
            md = json.loads(md)
        return md.get("_task_behavior_hash")
    except Exception:
        return None


def hashes(paths, sample=25):
    """Which generator produced these rows -- so a purge is verifiable, not hopeful."""
    seen = collections.Counter()
    for p in paths[:sample]:
        try:
            with open(p) as fh:
                line = fh.readline()
            md = json.loads(line).get("metadata") or {}
            if isinstance(md, str):
                md = json.loads(md)
            seen[md.get("_task_behavior_hash") or "NONE"] += 1
        except Exception:
            seen["UNREADABLE"] += 1
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="generated_data directory")
    ap.add_argument("--prefix", default="rc11.2-*")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--keep-hash", help="delete only rows NOT at this generator hash -- how you "
                                        "clean up after workers that still hold the old module")
    ap.add_argument("--apply", action="store_true", help="actually delete; default is a dry run")
    a = ap.parse_args()

    found = shard_files(a.root, a.prefix, a.tasks)
    if a.keep_hash:
        # Reads every file, not a sample: the point is to spare the rows that are already correct.
        kept = 0
        for task, paths in found.items():
            keep = [p for p in paths if (row_hash(p) or "").startswith(a.keep_hash)]
            kept += len(keep)
            found[task] = [p for p in paths if p not in set(keep)]
        print(f"[purge] keeping {kept:,} files already at {a.keep_hash}")
        found = {k: v for k, v in found.items() if v}
    if not found:
        return print(f"[purge] no files for {a.tasks} under {a.root}/{a.prefix}")
    total = 0
    for task in sorted(found):
        paths = found[task]
        total += len(paths)
        h = "  ".join(f"{k[:8]}x{v}" for k, v in hashes(paths).most_common())
        print(f"  {task:30} {len(paths):>7,} files   sampled hashes: {h}")
    if not a.apply:
        return print(f"[purge] DRY RUN: {total:,} files would be deleted. Re-run with --apply.")
    gone = 0
    for paths in found.values():
        for p in paths:
            try:
                os.unlink(p)
                gone += 1
            except OSError:
                pass
    print(f"[purge] deleted {gone:,}/{total:,} files. Running workers refill these slots; collect "
          f"re-pushes them on its own (new size/mtime => new file_key).")


if __name__ == "__main__":
    main()
