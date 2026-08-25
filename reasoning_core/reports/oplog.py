#!/usr/bin/env python3
"""Append-only journal of G5K operations, so placement knowledge is measured instead of remembered.

Every painful lesson this project keeps re-learning is a statistic someone already paid for: which
site actually admits jobs, how long besteffort really waits before it runs, which cluster silently
fails. `sweep` reads job state straight from OAR and records submit -> start -> end, so
`stats` can answer "where should I submit this" from data rather than from memory.

    python -m reasoning_core.reports.oplog sweep          # poll all sites, update the journal
    python -m reasoning_core.reports.oplog stats          # success rate + median wait per cluster
    python -m reasoning_core.reports.oplog notes          # refresh the block in .CLAUDE_NOTES.md

The journal is keyed site+job_id, so sweeping repeatedly is idempotent and finished jobs freeze.
"""
from __future__ import annotations
import argparse, json, os, re, statistics as st, subprocess, time
from collections import defaultdict
from pathlib import Path

SITES = ["lille", "grenoble", "rennes", "sophia", "nancy", "lyon"]
SSH = ["ssh", "-F", os.path.expanduser("~/.ssh/g5k_ssh_config"),
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
LOG = Path(__file__).with_name("build") / "oplog.jsonl"
NOTES = Path("/mnt/nfs_share_magnet2/dsileo/sandboxes/rc_grad/.CLAUDE_NOTES.md")
BEGIN, END = "<!-- OPLOG:BEGIN -->", "<!-- OPLOG:END -->"


def _run(site, cmd, timeout=60):
    try:
        r = subprocess.run(SSH + [f"{site}.g5k", cmd], capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _ts(s):
    try:
        return time.mktime(time.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return None


def _parse(blob):
    """oarstat -f blocks -> dicts. OAR prints 'id:' not 'Job_Id:' -- a real trap in this codebase."""
    jobs, cur = [], {}
    for line in blob.splitlines():
        if re.match(r"^Job_Id:|^id:", line.strip()):
            if cur:
                jobs.append(cur)
            cur = {"job": line.split(":", 1)[1].strip()}
        elif "=" in line and cur is not None:
            k, _, v = line.partition("=")
            cur[k.strip().lower()] = v.strip()
    if cur:
        jobs.append(cur)
    return jobs


def sweep(sites):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try:
                r = json.loads(line)
                seen[(r["site"], r["job"])] = r
            except Exception:
                pass
    n_new = n_upd = 0
    for site in sites:
        ids = [j["job"] for j in _parse(_run(site, "oarstat -u $USER -f"))]
        # also refresh anything we recorded but that had not finished
        ids += [k[1] for k, v in seen.items()
                if k[0] == site and v.get("state") not in ("Terminated", "Error")]
        for jid in sorted(set(ids)):
            j = _parse(_run(site, f"oarstat -f -j {jid}"))
            if not j:
                continue
            j = j[0]
            sub, sta, sto = (_ts(j.get(k, "")) for k in
                             ("submission_time", "start_time", "stop_time"))
            rec = {"site": site, "job": jid, "name": j.get("name", ""),
                   "queue": j.get("queue", ""), "state": j.get("state", ""),
                   "host": (j.get("assigned_hostnames", "") or "").split(".")[0],
                   "wanted": j.get("wanted_resources", "")[:120],
                   "submitted": sub, "started": sta, "stopped": sto,
                   "wait_s": (sta - sub) if (sta and sub and sta > sub) else None,
                   "run_s": (sto - sta) if (sto and sta and sto > sta) else None,
                   "exit": j.get("exit_code", ""), "seen": time.time()}
            key = (site, jid)
            if key not in seen:
                n_new += 1
            elif seen[key].get("state") != rec["state"]:
                n_upd += 1
            seen[key] = rec
    with LOG.open("w") as f:
        for r in sorted(seen.values(), key=lambda x: (x["site"], x["job"])):
            f.write(json.dumps(r) + "\n")
    print(f"[oplog] {len(seen)} jobs journaled ({n_new} new, {n_upd} updated) -> {LOG}")
    return list(seen.values())


def _cluster(r):
    h = r.get("host") or ""
    return re.sub(r"-?\d+$", "", h) or "(unscheduled)"


def _exit_ok(raw):
    """True when OAR's exit_code field means success. Accepts "0", "0 (0,0,0)" and an empty field
    (jobs that ended before OAR recorded a code)."""
    text = str(raw or "").strip()
    if not text:
        return True
    head = text.split()[0]
    try:
        return int(head) == 0
    except ValueError:
        return False


def _fmt(sec):
    if sec is None:
        return "-"
    return f"{sec/60:.0f}m" if sec < 5400 else f"{sec/3600:.1f}h"


def stats():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    if not rows:
        return print("[oplog] empty; run `sweep` first")
    by = defaultdict(list)
    for r in rows:
        by[(r["site"], _cluster(r), r["queue"] or "default")].append(r)
    print(f"{'site':10}{'cluster':14}{'queue':11}{'n':>3}{'started':>9}{'ok':>7}"
          f"{'med wait':>10}{'med run':>9}")
    for k, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        started = [r for r in rs if r.get("started")]
        # OAR journals exit_code as "0 (0,0,0)", not "0" -- an equality test against "0" scores every
        # successful job as a failure and prints a uniform 0% ok rate. Parse the leading integer.
        okd = [r for r in rs if r.get("state") == "Terminated" and _exit_ok(r.get("exit"))]
        w = [r["wait_s"] for r in started if r.get("wait_s") is not None]
        d = [r["run_s"] for r in rs if r.get("run_s") is not None]
        print(f"{k[0]:10}{k[1]:14}{k[2]:11}{len(rs):>3}{len(started)/len(rs):>8.0%}"
              f"{(len(okd)/len(rs)):>7.0%}{_fmt(st.median(w) if w else None):>10}"
              f"{_fmt(st.median(d) if d else None):>9}")
    return rows


def notes():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    by = defaultdict(list)
    for r in rows:
        by[(r["site"], _cluster(r), r["queue"] or "default")].append(r)
    lines = ["| site | cluster | queue | n | started | median wait |",
             "|---|---|---|---|---|---|"]
    for k, rs in sorted(by.items(), key=lambda x: -len(x[1]))[:12]:
        started = [r for r in rs if r.get("started")]
        w = [r["wait_s"] for r in started if r.get("wait_s") is not None]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | "
                     f"{len(started)/len(rs):.0%} | {_fmt(st.median(w) if w else None)} |")
    block = (f"{BEGIN}\n<!-- generated by reasoning_core.reports.oplog; do not hand-edit -->\n"
             f"## Where jobs actually run (measured, {time.strftime('%Y-%m-%d')})\n\n"
             + "\n".join(lines) + f"\n\n{END}")
    if not NOTES.exists():
        return print(f"[oplog] {NOTES} missing")
    txt = NOTES.read_text()
    if BEGIN in txt and END in txt:
        txt = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, txt, flags=re.S)
    else:
        txt = txt.rstrip() + "\n\n" + block + "\n"
    NOTES.write_text(txt)
    print(f"[oplog] refreshed placement table in {NOTES.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "stats", "notes"])
    ap.add_argument("--sites", nargs="+", default=SITES)
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep(a.sites)
    elif a.cmd == "stats":
        stats()
    else:
        notes()
