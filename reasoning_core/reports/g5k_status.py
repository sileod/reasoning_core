"""Poll every Grid'5000 site and emit one status snapshot.

OAR is per-site: `oarstat` run on lille reports nothing about jobs at grenoble, and both mount the
same storage. Anything that reports "the queue" from one frontend is wrong. This walks all sites.

    python -m reasoning_core.reports.g5k_status --out reports/build/status.json

Job kind is inferred from the script name in the OAR command, because the two pipelines are operated
separately and mixing them in one list hides whichever is smaller: `gpu` = influence/eval arms,
`gen` = the procedural-data generator array and its collect job.
"""
from __future__ import annotations
import argparse, json, re, subprocess, time
from pathlib import Path

SITES = ["lille", "grenoble", "rennes", "sophia", "nancy", "lyon"]
SSH = ["ssh", "-F", "/mnt/nfs_share_magnet2/dsileo/.ssh/g5k_ssh_config",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
ST = "/srv/storage/magnet@storage1.lille.grid5000.fr/dsileo"


def _run(site, cmd, timeout=45):
    try:
        r = subprocess.run(SSH + [f"{site}.g5k", cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _epoch(v):
    """OAR prints local wall-clock; the page shows both the stamp and an age, so parse to epoch."""
    try:
        return time.mktime(time.strptime(v.strip(), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return None


def kind(name, cmd):
    blob = f"{name} {cmd}".lower()
    if any(k in blob for k in ("rcgen", "generate", "collect", "launcher")):
        return "gen"
    if any(k in blob for k in ("rm_", "cog4", "wmix", "wssh", "influence", "pt360", "run_3b")):
        return "gpu"
    return "other"


def poll(site):
    # -f gives name + command; the default table does not, and the job kind lives in the command.
    out = _run(site, "oarstat -u $USER -f 2>/dev/null")
    jobs, cur = [], {}
    for line in out.splitlines():
        m = re.match(r"^id:\s*(\d+)", line.strip())
        if m:
            if cur:
                jobs.append(cur)
            cur = {"id": m.group(1), "site": site}
            continue
        m = re.match(r"^\s*(\w+)\s*=\s*(.*)$", line)
        if m and cur:
            k, v = m.group(1), m.group(2).strip()
            # oarstat -f emits snake_case: submissionTime never matches, which is why the
            # dashboard had no times at all. Same trap oplog.py hit with id: vs Job_Id:.
            if k in ("state", "name", "queue", "command", "submission_time", "start_time",
                     "walltime", "assigned_hostnames", "types", "stdout_file"):
                cur[k.lower()] = v
    if cur:
        jobs.append(cur)
    now = time.time()
    for j in jobs:
        for src, dst in (("submission_time", "submitted"), ("start_time", "started")):
            ts = _epoch(j.pop(src, ""))
            j[dst] = ts
            if ts:
                j[dst + "_h"] = round((now - ts) / 3600, 2)   # age, so the page needs no clock
        j["kind"] = kind(j.get("name", "") or j.get("stdout_file", ""), j.get("command", ""))
        h = j.get("assigned_hostnames", "")
        j["node"] = h.split(".")[0] if h else ""
        j["script"] = Path(j.get("command", "")).name
        j["state"] = j.get("state", "?")
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reasoning_core/reports/build/status.json")
    ap.add_argument("--sites", nargs="+", default=SITES)
    a = ap.parse_args()

    jobs = []
    for s in a.sites:
        jobs += poll(s)

    repo = Path(__file__).resolve().parents[2]
    cells = sorted(repo.glob("per_task_results/influence_COLL-*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    recent = [{"name": p.name, "age_h": round((time.time() - p.stat().st_mtime) / 3600, 1)}
              for p in cells[:12]]
    gen = _run("lille", f"ls -1t {ST}/libs/reasoning_core/reasoning_core/generated_data 2>/dev/null | head -8")

    snap = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": jobs,
        "by_state": {k: sum(1 for j in jobs if j["state"] == k) for k in {j["state"] for j in jobs}},
        "by_kind": {k: sum(1 for j in jobs if j["kind"] == k) for k in {j["kind"] for j in jobs}},
        "cells_total": len(cells),
        "recent_cells": recent,
        "generated_data": [x for x in gen.splitlines() if x.strip()],
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap, indent=1))
    print(f"[g5k] {len(jobs)} jobs across {len(a.sites)} sites -> {p}")
    for k, v in sorted(snap["by_state"].items()):
        print(f"   state {k}: {v}")


if __name__ == "__main__":
    main()
