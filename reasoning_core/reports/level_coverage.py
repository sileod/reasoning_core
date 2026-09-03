#!/usr/bin/env python3
"""Per-task level coverage, generation cost, and whether `level` actually does anything.

A build can look healthy (jobs Running, shard dirs growing) while a task emits NOTHING: the writer
creates the file before serialising, so a task that raises mid-write leaves a 0-byte .jsonl that is
indistinguishable from ordinary output. Counting files is therefore not a check; only rows are.

Beyond coverage this reports SENSITIVITY: median prompt tokens and median generation time at the
lowest vs highest realised level. A task whose prompt and time are flat across levels is not
responding to its difficulty knob, which is the defect that got evidence_sufficiency demoted.

    python -m reasoning_core.reports.level_coverage scan --root <generated_data> --out levels.json
    python -m reasoning_core.reports.level_coverage render --in levels.json --out build/levels.html
"""
from __future__ import annotations
import argparse, collections, json, os, random, glob, statistics as st, time

# generation_worker.py caps these tasks below the requested max, so a "missing" top level is by
# design, not a defect. Keep in sync with generation_worker.py:custom_max.
CUSTOM_MAX = {"proof_reconstruction": 2, "bayesian_association": 0, "bayesian_intervention": 0,
              "logic_nli": 3, "evidence_retrieval": 3,
              "table_conversion": 4}
FLAT = 1.15  # below this ratio low->high level, the knob is not moving the task


PROMPT_CAP = 400        # prompts kept per (task, level); enough for a stable rate, bounded memory
BOILER_SHARE = 0.9      # a line in >=90% of a cell's prompts is fixed text, not content


def _bylevel(d):
    out = collections.defaultdict(dict)
    for (t, l), v in d.items():
        if v is not None:
            out[t][str(l)] = v
    return {t: dict(sorted(v.items(), key=lambda x: int(x[0]))) for t, v in out.items()}


def _dup_and_boilerplate(texts):
    """Per (task, level): how often the generator repeats itself, and how much of a prompt is fixed.

    Boilerplate is measured on LINES present in nearly every prompt of the cell -- the instruction
    preamble a task restates every time. Under packing that text is paid for on every row while
    carrying no per-example signal, so a high ratio means the length is going to scaffolding rather
    than to the problem. Lines are counted over DISTINCT prompts, so duplicates cannot inflate it.
    """
    dup, boiler = {}, {}
    for key, ps in texts.items():
        if len(ps) < 2:
            continue
        uniq = set(ps)
        dup[key] = 1 - len(uniq) / len(ps)
        seen = collections.Counter()
        for u in uniq:
            for ln in set(u.splitlines()):
                seen[ln] += 1
        fixed = {ln for ln, c in seen.items() if ln.strip() and c >= BOILER_SHARE * len(uniq)}
        tot = fix = 0
        for u in uniq:
            for ln in u.splitlines():
                w = len(ln.split())
                tot += w
                if ln in fixed:
                    fix += w
        boiler[key] = (fix / tot) if tot else None
    return dup, boiler


def scan(root, prefix, per_task, seed=0):
    rng = random.Random(seed)
    tally, empty, files = collections.Counter(), collections.Counter(), collections.Counter()
    tok = collections.defaultdict(list); tim = collections.defaultdict(list)
    texts = collections.defaultdict(list)          # (task, level) -> prompts, for dup/boilerplate
    dirs = sorted(glob.glob(os.path.join(root, prefix)))
    for d in dirs:
        by = collections.defaultdict(list)
        for f in os.listdir(d):
            if f.endswith(".jsonl"):
                by[f.rsplit("-", 1)[0]].append(f)
        for task, fs in by.items():
            files[task] += len(fs)
            for f in rng.sample(fs, min(per_task, len(fs))):
                p = os.path.join(d, f)
                if os.path.getsize(p) == 0:
                    empty[task] += 1
                    continue
                with open(p) as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            m = row["metadata"]
                            m = json.loads(m) if isinstance(m, str) else m
                        except Exception:
                            continue
                        t, lv = m.get("_task", task), m.get("_level")
                        tally[(t, lv)] += 1
                        if row.get("prompt") and len(texts[(t, lv)]) < PROMPT_CAP:
                            texts[(t, lv)].append(row["prompt"])
                        if isinstance(m.get("_prompt_tokens"), (int, float)):
                            tok[(t, lv)].append(float(m["_prompt_tokens"]))
                        if isinstance(m.get("_time"), (int, float)):
                            tim[(t, lv)].append(float(m["_time"]))
    per = collections.defaultdict(dict)
    for (t, l), c in tally.items():
        per[t][str(l)] = c
    def med(d):
        out = collections.defaultdict(dict)
        for (t, l), v in d.items():
            if v: out[t][str(l)] = st.median(v)
        return {t: dict(sorted(v.items(), key=lambda x: int(x[0]))) for t, v in out.items()}
    dup, boiler = _dup_and_boilerplate(texts)
    return {"dirs": [os.path.basename(x) for x in dirs], "sampled_per_task_per_dir": per_task,
            "dup_by_level": _bylevel(dup), "boiler_by_level": _bylevel(boiler),
            "levels": {t: dict(sorted(v.items(), key=lambda x: int(x[0]))) for t, v in per.items()},
            "files": dict(files), "empty_sampled": dict(empty),
            "tokens_by_level": med(tok), "time_by_level": med(tim)}


def batchcost(root, prefix, since=None):
    """True per-row cost from batches.jsonl: total batch wall time over rows actually delivered.

    Every shard dir appends to ONE batches.jsonl at the root, so a rebuild's lines sit next to the
    previous release's, and the log carries no shard name to separate them. Pass `since` (epoch, or
    "YYYY-MM-DD HH:MM:SS" -- the release's submission time) to cut the earlier build out; without it
    the cost is averaged over two different generators and is meaningless.

    Do NOT default this to a directory timestamp: a shard dir under active write has its ctime and
    mtime bumped by every new file, so it reads as "seconds ago" and drops the whole run.
    """
    if isinstance(since, str):
        since = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M:%S"))
    if since is None:
        print("[batchcost] WARNING: no --since; mixing every release present in batches.jsonl")
        since = 0
    tot = collections.defaultdict(lambda: {"t": 0.0, "rows": 0, "t_bad": 0.0, "n": 0, "bad": 0})
    path = os.path.join(root, "batches.jsonl")
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                b = json.loads(line)
            except Exception:
                continue                      # concurrent appends can interleave a torn line
            if b.get("ts", 0) < since:
                continue
            r = tot[b["task"]]
            r["t"] += b.get("batch_time_s", 0.0); r["rows"] += b.get("rows", 0); r["n"] += 1
            if not b.get("rows"):
                r["t_bad"] += b.get("batch_time_s", 0.0); r["bad"] += 1
    return {k: {"s_per_row": v["t"] / v["rows"] if v["rows"] else None,
                "waste": v["t_bad"] / v["t"] if v["t"] else None,
                "batches": v["n"], "empty_batches": v["bad"], "rows": v["rows"]}
            for k, v in tot.items()}


def _ratio(by_level):
    """Median at the highest realised level divided by the lowest. None if <2 levels."""
    ks = sorted(by_level, key=int)
    if len(ks) < 2 or not by_level[ks[0]]:
        return None
    return by_level[ks[-1]] / by_level[ks[0]]


ZS_MODEL = "meta/llama-3.3-70b-instruct"   # the model the "70B solve" column names


def summarize(data, want, zs=None, bc=None):
    zs = zs or {}
    bc = bc or {}
    rows = []
    for task in sorted(set(data["files"]) | set(data["levels"])):
        got = data["levels"].get(task, {})
        n = sum(got.values())
        cap = CUSTOM_MAX.get(task)
        expect = [l for l in want if cap is None or l <= cap]
        missing = [l for l in expect if not got.get(str(l))]
        capped = [l for l in want if cap is not None and l > cap]
        tk = data.get("tokens_by_level", {}).get(task, {})
        tm = data.get("time_by_level", {}).get(task, {})
        med_ms = (st.median(list(tm.values())) * 1000) if tm else None
        r_tok, r_tim = _ratio(tk), _ratio(tm)
        flat = (r_tok is not None and r_tok < FLAT) and (r_tim is None or r_tim < FLAT)
        status = ("DEAD" if n == 0 else "gaps" if missing else
                  "flat" if flat else "capped" if capped else "ok")
        rows.append({"task": task, "n": n, "files": data["files"].get(task, 0), "status": status,
                     "levels": got, "missing": missing, "capped": capped, "med_ms": med_ms,
                     "r_tok": r_tok, "r_tim": r_tim,
                     "cost": (bc.get(task) or {}).get("s_per_row"),
                     "waste": (bc.get(task) or {}).get("waste"),
                     # Only this model: the column says "70B solve" and means it. The probe
                     # cache is keyed task|level|model and also holds deepseek-v4-flash cells,
                     # which measure something else entirely (answering with no reasoning at
                     # all). Cells predating the model key were all 70B, hence the default.
                     "solve": {int(k.split("|")[1]): v["solve_rate"]
                               for k, v in zs.items()
                               if k.split("|")[0] == task and v.get("status") == "ok"
                               and v.get("model", ZS_MODEL) == ZS_MODEL},
                     "dup": {int(k): v for k, v in
                             (data.get("dup_by_level", {}).get(task) or {}).items()},
                     "boiler": {int(k): v for k, v in
                                (data.get("boiler_by_level", {}).get(task) or {}).items()},
                     "tok_lo": (list(tk.values())[0] if tk else None),
                     "tok_hi": (list(tk.values())[-1] if tk else None)})
    order = {"DEAD": 0, "flat": 1, "gaps": 2, "capped": 3, "ok": 4}
    rows.sort(key=lambda r: (order[r["status"]], r["task"]))
    return rows


def _qshade(vals):
    """Quantile rank -> 0..4 shade. Rank-based so a few extreme tasks cannot flatten the scale."""
    order = sorted(v for v in vals if v is not None)
    def shade(v):
        if v is None or not order: return ""
        i = sum(1 for x in order if x < v) / max(len(order) - 1, 1)
        return f" q{min(int(i * 5), 4)}"
    return shade


def solve_cells(sv):
    """Per-level solve rate, then the slope. Falling solve rate as level rises is the only direct
    evidence the knob makes problems HARDER rather than merely longer."""
    if not sv:
        return '<td class=n>&mdash;</td><td class=n>&mdash;</td>'
    ks = sorted(sv)
    series = " ".join(f"{sv[k]:.0%}" for k in ks)
    drop = sv[ks[0]] - sv[ks[-1]] if len(ks) > 1 else None
    if drop is None:
        cell = '<td class=n>&mdash;</td>'
    elif drop > 0.05:
        cell = f'<td class="n harder" data-v="{drop:.4f}">-{drop:.0%}</td>'   # harder: expected
    elif drop < -0.05:
        cell = f'<td class="n easier" data-v="{drop:.4f}">+{-drop:.0%}</td>'  # EASIER with level
    else:
        cell = f'<td class="n flatv" data-v="{drop:.4f}">flat</td>'
    return f'<td class="n sv" data-v="{sv[ks[0]]:.4f}">{series}</td>' + cell


SORT_JS = """<script>
(function(){
  const tb = document.querySelector('table'), body = tb.tBodies[0];
  const hs = [...tb.rows[0].cells];
  // data-v carries the numeric truth behind a formatted cell ("2 1 3 1 1", "-12%", "1.4x");
  // without it a text sort would order 100 before 20.
  const key = c => c.dataset.v !== undefined ? parseFloat(c.dataset.v)
                 : (c.textContent.trim() === '\u2014' ? null : c.textContent.trim());
  hs.forEach((h, i) => h.onclick = () => {
    const dir = h.dataset.dir === 'asc' ? -1 : 1;
    hs.forEach(x => { delete x.dataset.dir; x.textContent = x.textContent.replace(/ [\u25b2\u25bc]$/, ''); });
    h.dataset.dir = dir > 0 ? 'asc' : 'desc';
    h.textContent += dir > 0 ? ' \u25b2' : ' \u25bc';
    [...body.rows].slice(1).sort((a, b) => {
      const x = key(a.cells[i]), y = key(b.cells[i]);
      if (x === null) return 1;                       // blanks sink, whichever way you sort
      if (y === null) return -1;
      return (typeof x === 'number' ? x - y : String(x).localeCompare(String(y))) * dir;
    }).forEach(r => body.appendChild(r));
  });
})();
</script>"""


def pct_cells(series, warn):
    """Per-level percentages plus the worst one, so a bad level cannot hide behind a good average."""
    if not series:
        return '<td class=n>&mdash;</td><td class=n>&mdash;</td>'
    ks = sorted(series)
    body = " ".join(f"{series[k]:.0%}" for k in ks)
    hi = max(series.values())
    cls = "n flatv" if hi >= warn else "n"
    return (f'<td class="n sv" data-v="{hi:.4f}">{body}</td>'
            f'<td class="{cls}" data-v="{hi:.4f}">{hi:.0%}</td>')


def render(data, want, out, zs=None, bc=None):
    rows = summarize(data, want, zs, bc)
    tot = collections.Counter()
    for r in rows:
        for l, c in r["levels"].items(): tot[int(l)] += c
    grand = sum(tot.values()) or 1
    peak = max(tot.values()) if tot else 1
    bar = "".join(f'<div class=b><span>L{l}</span><i style="width:{tot.get(l,0)/peak*100:.1f}%"></i>'
                  f'<b>{tot.get(l,0)/grand:.1%}</b></div>' for l in want)
    sh_ms = _qshade([r["med_ms"] for r in rows])       # shade on SLOWNESS, so dark = slow
    sh_cost = _qshade([r["cost"] for r in rows])
    srank = {"DEAD": 0, "flat": 1, "gaps": 2, "capped": 3, "ok": 4}

    def rat(v):
        if v is None: return '<td class=n>&mdash;</td>'
        cls = " flatv" if v < FLAT else ""
        return f'<td class="n{cls}" data-v="{v:.4f}">{v:.2f}&times;</td>'

    def rate(r):
        """Accepted-row throughput, then the one that also pays for rejected candidates."""
        a = 1000 / r["med_ms"] if r.get("med_ms") else None
        b = 1 / r["cost"] if r.get("cost") else None
        thin = a and b and b < a / 4       # most of the wall clock went to discarded candidates
        return ((f'<td class="n ms{sh_ms(r["med_ms"])}" data-v="{a:.4f}">{a:,.1f}</td>'
                 if a else '<td class=n>&mdash;</td>')
                + (f'<td class="n ms{sh_cost(r["cost"])}{" flatv" if thin else ""}" '
                   f'data-v="{b:.4f}">{b:,.2f}</td>' if b else '<td class=n>&mdash;</td>'))

    body = "".join(
        f'<tr class="{r["status"]}"><td class=t>{r["task"]}</td>'
        f'<td class=s data-v="{srank[r["status"]]}">{r["status"]}</td>'
        f'<td class=n data-v="{r["n"]}">{r["n"]:,}</td>'
        + rate(r)
        + (f'<td class=n data-v="{r["tok_hi"]:.0f}">{r["tok_lo"]:,.0f}&rarr;{r["tok_hi"]:,.0f}</td>'
           if r["tok_lo"] is not None else '<td class=n>&mdash;</td>')
        + rat(r["r_tok"]) + rat(r["r_tim"])
        + pct_cells(r["dup"], 0.05) + pct_cells(r["boiler"], 0.50)
        + solve_cells(r["solve"])
        + f'<td class=m>{" ".join("L%d" % l for l in r["missing"]) or "&mdash;"}</td></tr>'
        for r in rows)
    c = collections.Counter(r["status"] for r in rows)
    rel = (data.get("dirs") or ["?"])[0].rsplit("-", 1)[0]
    html = f"""<title>Task Scaling</title>
<style>
:root{{--bg:#f6f7f9;--pa:#fff;--ln:#dcdfe6;--ink:#1b1f27;--i2:#5a6273;--i3:#878fa1;
--ok:#1f8a5a;--wa:#a8781a;--de:#b04a44;--mo:ui-monospace,Menlo,Consolas,monospace;
--q0:#eef2f7;--q1:#d3dfec;--q2:#adc3dd;--q3:#7d9dc4;--q4:#4d73a4;--qink:#10161f}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#12151a;--pa:#181c23;
--ln:#2b313b;--ink:#e6e9ef;--i2:#a2abbb;--i3:#6f7787;--ok:#4fc98c;--wa:#d7a63f;--de:#e58079;
--q0:#1b2028;--q1:#22303f;--q2:#2c4257;--q3:#3a5b78;--q4:#4d789e;--qink:#eaf1f8}}}}
:root[data-theme=dark]{{--bg:#12151a;--pa:#181c23;--ln:#2b313b;--ink:#e6e9ef;--i2:#a2abbb;
--i3:#6f7787;--ok:#4fc98c;--wa:#d7a63f;--de:#e58079;
--q0:#1b2028;--q1:#22303f;--q2:#2c4257;--q3:#3a5b78;--q4:#4d789e;--qink:#eaf1f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}
.w{{max-width:1080px;margin:0 auto;padding:28px 20px 64px;display:flex;flex-direction:column;gap:20px}}
h1{{margin:0;font-size:19px}}.sub{{color:var(--i3);font:12px/1.5 var(--mo)}}
section{{background:var(--pa);border:1px solid var(--ln);border-radius:8px;overflow:hidden}}
h2{{margin:0;padding:11px 14px;font-size:12px;text-transform:uppercase;letter-spacing:.07em;
color:var(--i2);border-bottom:1px solid var(--ln);font-weight:600}}
.bars{{padding:14px}}.b{{display:flex;align-items:center;gap:10px;margin:5px 0;font:12px var(--mo)}}
.b span{{width:26px;color:var(--i2)}}.b i{{height:11px;background:var(--i2);border-radius:2px;min-width:2px}}
.b b{{color:var(--i3);font-weight:400}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{text-align:left;padding:6px 12px;border-bottom:1px solid var(--ln);white-space:nowrap}}
th{{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--i3);
cursor:pointer;user-select:none}}th:hover{{color:var(--ink)}}
td.n{{font-family:var(--mo);font-variant-numeric:tabular-nums;color:var(--i2);text-align:right}}
td.t{{font-family:var(--mo)}}td.m{{font-family:var(--mo);color:var(--wa)}}
td.s{{font-weight:600}}tr.ok td.s{{color:var(--ok)}}tr.gaps td.s{{color:var(--wa)}}
tr.capped td.s{{color:var(--i2)}}tr.flat td.s{{color:var(--wa)}}
tr.DEAD td.s,tr.DEAD td.t{{color:var(--de)}}
td.ms{{color:var(--qink)}}td.q0{{background:var(--q0)}}td.q1{{background:var(--q1)}}
td.q2{{background:var(--q2)}}td.q3{{background:var(--q3)}}td.q4{{background:var(--q4)}}
td.flatv{{color:var(--wa);font-weight:600}}
td.sv{{color:var(--i2);letter-spacing:.04em}}
td.harder{{color:var(--ok);font-weight:600}}td.easier{{color:var(--de);font-weight:600}}
.key{{padding:10px 14px;color:var(--i3);font:11.5px/1.6 var(--mo);border-top:1px solid var(--ln)}}
</style>
<div class=w>
<div><h1>Task scaling</h1><div class=sub>{rel} &middot; {len(data['dirs'])} shard dirs &middot; {grand:,} rows
sampled &middot; {len(rows)} tasks &middot; {c.get('DEAD',0)} dead, {c.get('flat',0)} flat,
{c.get('gaps',0)} with gaps, {c.get('capped',0)} intentionally capped</div></div>
<section><h2>Rows per level (all tasks)</h2><div class=bars>{bar}</div></section>
<section><h2>Per task</h2><div class=scroll><table>
<tr><th>task</th><th>status</th><th>rows</th><th>ex/s</th><th>ex/s (batch)</th>
<th>prompt tok lo&rarr;hi</th>
<th>tok ratio</th><th>time ratio</th><th>dup by level</th><th>dup max</th>
<th>boilerplate by level</th><th>boiler max</th><th>70B solve by level</th><th>solve drop</th>
<th>missing</th></tr>
{body}</table></div>{SORT_JS}
<div class=key>Click any column to sort. ex/s = accepted rows per second, from the median _time of an
ACCEPTED row. ex/s (batch) = rows actually DELIVERED per second of batch wall time, from batches.jsonl,
so it also pays for rejected candidates and failed batches; marked when it falls below a quarter of
ex/s, meaning most of the clock went to candidates that were thrown away. Both shaded by quantile
rank across tasks (dark = slow), so a few very slow tasks cannot flatten the scale. ratio = median at highest realised level / lowest; &lt;{FLAT} is marked, meaning the level
knob is not moving that task. capped = generation_worker caps the level on purpose. 70B solve =
dup = share of sampled prompts that repeat within a level; above 5% is marked. boilerplate = share
of prompt words on lines present in &ge;90% of that level's DISTINCT prompts, i.e. the fixed
instruction text restated on every row -- above 50% is marked, meaning most of the length is
scaffolding rather than problem. 70B solve = zero-shot solve rate of llama-3.3-70b-instruct per level (low to high), n=2, scored by the task's
own score_answer. solve drop = first level minus last: a DROP is the knob working, "flat" means the
levels differ in size but not difficulty, and a rise (red) means high levels are EASIER.</div>
</section>
</div>"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(html)
    return out, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--root", required=True)
    s.add_argument("--prefix", default="rc11.0-*"); s.add_argument("--per-task", type=int, default=8)
    s.add_argument("--out", required=True)
    b = sub.add_parser("batchcost"); b.add_argument("--root", required=True)
    b.add_argument("--prefix", default="rc11.2-*"); b.add_argument("--out", required=True)
    b.add_argument("--since", default=None,
                   help='cutoff: epoch or "YYYY-MM-DD HH:MM:SS" (the release submission time)')
    r = sub.add_parser("render"); r.add_argument("--in", dest="inp", required=True)
    r.add_argument("--out", default="reasoning_core/reports/build/levels.html")
    r.add_argument("--zeroshot", default="reasoning_core/reports/build/zeroshot.json")
    r.add_argument("--batchcost", default="/tmp/batchcost.json")
    for p in (s, r): p.add_argument("--levels", default="0,1,2,3,4,5,6")
    a = ap.parse_args()                       # batchcost takes no --levels
    want = [int(x) for x in getattr(a, "levels", "0,1,2,3,4,5,6").split(",")]
    if a.cmd == "batchcost":
        since = a.since
        if since is not None and " " not in str(since):
            since = float(since)
        d = batchcost(a.root, a.prefix, since); json.dump(d, open(a.out, "w"))
        pr = [v["s_per_row"] for v in d.values() if v["s_per_row"]]
        print(f"[batchcost] {len(d)} tasks, {sum(v['batches'] for v in d.values()):,} batches, "
              f"median {st.median(pr) * 1000:,.0f} ms/row -> {a.out}" if pr else "[batchcost] no rows yet")
    elif a.cmd == "scan":
        d = scan(a.root, a.prefix, a.per_task); json.dump(d, open(a.out, "w"))
        print(f"[levels] {sum(sum(v.values()) for v in d['levels'].values()):,} rows, "
              f"{len(d['levels'])} tasks -> {a.out}")
    else:
        d = json.load(open(a.inp))
        zs = json.load(open(a.zeroshot)) if os.path.exists(a.zeroshot) else None
        bc = json.load(open(a.batchcost)) if os.path.exists(a.batchcost) else None
        out, rw = render(d, want, a.out, zs, bc)
        c = collections.Counter(x["status"] for x in rw)
        print(f"[levels] {out}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
