"""Rebuild the Atlas data block from per_task_results, with mixed protocols reconciled.

Other agents: this is the ONE place the Atlas matrix is derived. Edit the HTML by hand and the next
rebuild overwrites you; add a source here instead.

    python -m reasoning_core.reports.atlas_data --out atlas_data.json
    python -m reasoning_core.reports.atlas_data --patch-html path/to/margin_atlas.html

Every source declares its protocol. Standard sources (T300/M20) are used as-is; warm sources
(T75/M80) are mapped onto the standard axis by `protocol_calibration`, because the warm protocol
applies a per-leg gain of median 1.38x and mixing raw warm rows into a standard matrix overstates
them. Rows carry `p` ("std"/"warm") and `Lraw` so a reader can always recover what was measured.
"""
from __future__ import annotations
import argparse, collections, datetime, glob, json, re
from pathlib import Path

from reasoning_core.reports.protocol_calibration import fit, apply_row, uncalibrated_legs

PR = Path(__file__).resolve().parents[2] / "per_task_results"
LEGS = ['arc_challenge', 'arc_easy', 'balanced_copa', 'bbh_dev', 'bbh_dev_nomenu', 'bbh_test',
        'bbh_test_nomenu', 'blimp', 'commonsenseqa', 'folio', 'hs2_short', 'mmlu_logic',
        'mmlu_math', 'mmlu_math_macro', 'mmlu_math_nomenu', 'mmlu_other_cloze', 'openbookqa',
        'python_dpo', 'sciq', 'themis_py', 'ts_dpo', 'tulu3_short', 'uf_short', 'ultramix_short',
        'winogrande']

# collection, protocol, glob. `roster`/`rc` cells are rc-side; `rgym` cells are rg-side.
SOURCES = [
    ("rc", "std",  "influence_COLL-roster_RCV8SH*_S4[3456]_*_T300_M20_*.json"),
    ("rg", "std",  "influence_COLL-rgym_RGV8SH*_S5[1234]_*_T300_M20_*.json"),
    ("rc", "warm", "influence_COLL-roster_NEWV8_*_T75_M80_*.json"),
    ("rc", "warm", "influence_COLL-roster_PR47V8_*_T75_M80_*.json"),
]
# the tasks measured under BOTH protocols -- these fit the calibration
CAL_WARM = "influence_COLL-roster_W[LS]SH*_T75_M80_*.json"
CAL_STD = "influence_COLL-roster_RCV8SH*_S4[3456]_*_T300_M20_*.json"


def read(pattern):
    """{task: {leg: seed-averaged margin delta}} over every cell matching the glob."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(str(PR / pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for task, m in (d.get("tasks") or {}).items():
            for leg in LEGS:
                v = m.get(leg + "_mc_cloze_margin_delta")
                if isinstance(v, (int, float)):
                    acc[task][leg].append(v)
    return {t: {l: sum(v) / len(v) for l, v in d.items()} for t, d in acc.items()}


CELL = re.compile(r"influence_COLL-([^_]+)_(.+)_S\d+_T(\d+)_M(\d+)_([^_]+)_(\w+)\.json$")


def survey():
    """Every protocol on disk, keyed by what the filename records. No file reads.

    SOURCES stays a deliberate inclusion list -- protocols that share no calibration cannot
    be merged onto one axis, and guessing is worse than omitting. But a campaign matching no
    source used to vanish without a word, so report what exists and what the atlas covers.
    """
    covered = {Path(f).name for _, _, pattern in SOURCES for f in glob.glob(str(PR / pattern))}
    groups = collections.defaultdict(lambda: {"cells": 0, "shown": 0, "tags": set(), "mtime": 0})
    for path in PR.glob("influence_COLL-*.json"):
        match = CELL.match(path.name)
        if not match:
            continue
        coll, tag, steps, mix, main, init = match.groups()
        group = groups[(coll, main, f"T{steps}", f"M{mix}", init)]
        group["cells"] += 1
        group["shown"] += path.name in covered
        group["tags"].add(tag.split("_")[0])
        group["mtime"] = max(group["mtime"], path.stat().st_mtime)
    return groups


def report_uncovered(minimum=5, show=5):
    """Name the protocols the atlas is not showing, newest first.

    Ordered by recency, not size: most uncovered protocols are old sweeps at other model
    scales that correctly belong on their own axis. The one worth acting on is the campaign
    that just finished, so it goes at the top and the long tail stays a count.
    """
    missing = [(key, g) for key, g in survey().items() if not g["shown"] and g["cells"] >= minimum]
    if not missing:
        return
    missing.sort(key=lambda kv: -kv[1]["mtime"])
    print(f"[atlas] {len(missing)} protocol(s) on disk that no SOURCES entry covers, newest first:")
    for key, group in missing[:show]:
        stamp = datetime.date.fromtimestamp(group["mtime"]).isoformat()
        tags = ", ".join(sorted(group["tags"])[:4])
        print(f"[atlas]   {' '.join(key):46} {group['cells']:4d} cells  {stamp}  {tags}")
    if len(missing) > show:
        print(f"[atlas]   ... and {len(missing) - show} older protocol(s)")
    print("[atlas] add a SOURCES entry (plus a calibration, if the protocol differs) to include one.")


def build():
    cal = fit(read(CAL_STD), read(CAL_WARM))
    rows, seen = [], {}
    for coll, proto, pattern in SOURCES:
        for task, vals in read(pattern).items():
            if task in seen:                      # first source wins; std is listed first on purpose
                continue
            seen[task] = True
            row = {"t": task, "c": coll, "p": proto,
                   "L": apply_row(cal, vals) if proto == "warm" else dict(vals)}
            if proto == "warm":
                row["Lraw"] = dict(vals)
            rows.append(row)
    rows.sort(key=lambda r: r["t"])
    return {"rows": rows,
            "calibration": {l: {k: round(v, 6) if isinstance(v, float) else v
                                for k, v in c.items()} for l, c in cal.items()},
            "uncalibrated": uncalibrated_legs(cal, LEGS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--patch-html", help="replace the `const D = {...};` block in this file in place")
    a = ap.parse_args()
    D = build()
    warm = [r["t"] for r in D["rows"] if r["p"] == "warm"]
    print(f"[atlas] {len(D['rows'])} rows ({len(warm)} warm-protocol, calibrated onto the standard axis)")
    print(f"[atlas] calibrated legs {len(D['calibration'])}, pass-through {D['uncalibrated']}")
    report_uncovered()
    blob = json.dumps(D, separators=(",", ":"))
    if a.out:
        Path(a.out).write_text(blob)
        print(f"[atlas] -> {a.out}")
    if a.patch_html:
        p = Path(a.patch_html)
        t = p.read_text()
        m = re.search(r"const D = (\{.*?\});\n", t, re.S)
        if not m:
            raise SystemExit("[atlas] could not locate a single `const D = {...};` block")
        # MERGE, never replace. The block also carries `legs`, `pc` and `ax` (the varimax component
        # loadings and axis labels), which are produced by a different pipeline; a blind overwrite
        # silently deletes the whole principal-components section.
        existing = json.loads(m.group(1))
        keep = {k: v for k, v in existing.items() if k != "rows"}
        merged = json.dumps({**keep, **D}, separators=(",", ":"))
        p.write_text(t[:m.start()] + "const D = " + merged + ";\n" + t[m.end():])
        print(f"[atlas] patched {p}; preserved keys {sorted(keep)}")


if __name__ == "__main__":
    main()
