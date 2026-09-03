#!/usr/bin/env python3
"""Which task edits have never been measured.

Every manifest we write records the `behavior_hash` (docstring-stripped AST sha1) each task's rows
were generated from, so the evidence for a task is stale exactly when no manifest carries its CURRENT
hash. That turns "did I re-measure this after editing it?" into a lookup instead of a memory exercise.

    python -m reasoning_core.reports.task_staleness           # stale + never-measured
    python -m reasoning_core.reports.task_staleness --all     # every live task
    python -m reasoning_core.reports.task_staleness --names   # bare names, to feed a rebuild

STALE means the evidence predates the edit, not that the edit MATTERED -- `roster_drift` answers that
second question, on regenerated rows. A `*` marks an uncommitted file, whose hash nobody else can
reproduce yet.
"""
from __future__ import annotations
import argparse, ast, functools, hashlib, json, subprocess, time
from collections import defaultdict
from pathlib import Path

from reasoning_core.template import _strip_docstrings

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "reasoning_core" / "tasks"
# every place a generated-rows manifest lands: shared cache, ad-hoc carve dirs, sweep results
MANIFESTS = ["task_diagnostics/cache/task_rows/*/manifest.json", "*/*/manifest.json",
             "experiments/results/**/*.manifest.json"]
RANK = {"NEW": 0, "STALE": 1, "ok": 2}


@functools.lru_cache(maxsize=None)
def file_hash(path):
    # Many tasks share one module file; memoize so each module is parsed once, not once per task.
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    return hashlib.sha1(
        ast.dump(_strip_docstrings(tree), include_attributes=False).encode()).hexdigest()[:16]


def live(dev=False, generated=False):
    """task -> (current hash, source path, is_dev). Registry lookup only; nothing is instantiated.

    DevTasks are excluded from `list_tasks()` and so never generated, so they are off by default;
    `--dev` includes them, flagged, for the rare case of auditing one before promoting it. Landed
    task-search output is off for the same reason and opts in the same way: a probe measuring one
    needs its hash to know whether a cached cell still describes it."""
    import reasoning_core as rc
    roster = set(rc.list_tasks(include_generated=generated))
    names = roster | (set(getattr(rc, "DEV_DATASETS", {})) if dev else set())
    out = {}
    for name in sorted(names):
        src = rc.DATASETS.get(name) or getattr(rc, "DEV_DATASETS", {}).get(name)
        mod = getattr(src, "module_name", "")
        p = TASKS.joinpath(*mod.split(".")).with_suffix(".py")
        if p.exists():
            out[name] = (file_hash(str(p)), p, name not in roster)
    return out


def evidence():
    """task -> {hash: [newest mtime, n manifests]}, over both manifest shapes we have written."""
    ev = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for pat in MANIFESTS:
        for f in ROOT.glob(pat):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            pairs = list((d.get("behavior_hashes") or {}).items())
            if isinstance(d.get("tasks"), dict):          # sweep manifests nest it under each task
                pairs += [(t, v.get("behavior_hash")) for t, v in d["tasks"].items()
                          if isinstance(v, dict)]
            ts = _stamp(d.get("generated_at")) or f.stat().st_mtime
            for t, h in pairs:
                if h:
                    slot = ev[t][h]
                    slot[0], slot[1] = max(slot[0], ts), slot[1] + 1
    return ev


def _stamp(iso):
    """Manifests that record their own build time beat mtime, which a later rsync can rewrite."""
    try:
        return time.mktime(time.strptime(str(iso)[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None


def uncommitted():
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(TASKS)],
                         capture_output=True, text=True).stdout
    return {ROOT / l[3:].split(" -> ")[-1].strip() for l in out.splitlines() if l[3:].strip()}


def scan(dev=False):
    ev, dirty = evidence(), uncommitted()
    rows = []
    for name, (h, p, is_dev) in live(dev=dev).items():
        seen = ev.get(name, {})
        n_live = seen.get(h, (0, 0))[1]
        best = max(seen.items(), key=lambda kv: kv[1][0], default=None)
        rows.append({"task": name, "live": h, "n_at_live": n_live, "path": str(p), "dev": is_dev,
                     "status": "ok" if n_live else ("STALE" if seen else "NEW"),
                     "newest_hash": best[0] if best else None,
                     "newest_ts": best[1][0] if best else None,
                     "edited": p.stat().st_mtime, "dirty": p in dirty})
    rows.sort(key=lambda r: (RANK[r["status"]], -r["edited"]))
    return rows


def _day(ts):
    return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "-"


def render(rows, show_all=False):
    sel = [r for r in rows if show_all or r["status"] != "ok"]
    print(f"{'status':7}{'task':32}{'live':18}{'n':>3}  {'newest evidence':21}edited")
    for r in sel:
        ev = "-" if not r["newest_ts"] else (
            f"{_day(r['newest_ts'])} @{'=' if r['newest_hash'] == r['live'] else r['newest_hash'][:8]}")
        tag = ('*' if r['dirty'] else '') + (' [dev]' if r['dev'] else '')
        print(f"{r['status']:7}{r['task'] + tag:32}{r['live']:18}"
              f"{r['n_at_live']:>3}  {ev:21}{_day(r['edited'])}")
    n = defaultdict(int)
    for r in rows:
        n[r["status"]] += 1
    nd = sum(1 for r in rows if r["dev"])
    print(f"\n{len(rows)} roster tasks: {n['ok']} measured at current hash, {n['STALE']} stale, "
          f"{n['NEW']} never measured" + (f" ({nd} dev-only shown)" if nd else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include tasks whose evidence is current")
    ap.add_argument("--names", action="store_true", help="print stale+new names only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dev", action="store_true", help="also list DevTasks, which are never generated")
    a = ap.parse_args()
    rows = scan(dev=a.dev)
    if a.names:
        print(" ".join(r["task"] for r in rows if r["status"] != "ok" or a.all))
    elif a.json:
        print(json.dumps(rows, indent=1))
    else:
        render(rows, a.all)
