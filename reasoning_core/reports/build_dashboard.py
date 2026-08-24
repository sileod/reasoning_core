"""Render the G5K status snapshot as one self-contained page.

Two delivery paths, one file. Served over Tailscale it re-fetches `status.json` next to itself on
demand, so Refresh gets live data. Published as a Claude artifact the CSP blocks that fetch, so the
snapshot embedded at build time is what shows and the page says so rather than pretending.

    python -m reasoning_core.reports.g5k_status --out reasoning_core/reports/build/status.json
    python -m reasoning_core.reports.build_dashboard
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HTML = """<title>Grid'5000 Queue</title>
<style>
:root{
  --ground:#f5f6f8; --panel:#ffffff; --line:#dcdfe6; --line-soft:#eceef2;
  --ink:#1b1f27; --ink-2:#5a6273; --ink-3:#878fa1;
  --accent:#3d5a8a;
  --run:#1f8a5a; --run-bg:#e3f3ea;
  --wait:#a8781a; --wait-bg:#faf0d9;
  --dead:#b04a44; --dead-bg:#f8e6e4;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:ui-sans-serif,system-ui,"Segoe UI",Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#12151a; --panel:#181c23; --line:#2b313b; --line-soft:#222831;
  --ink:#e6e9ef; --ink-2:#a2abbb; --ink-3:#6f7787;
  --accent:#8fb0e0;
  --run:#4fc98c; --run-bg:#14301f;
  --wait:#d7a63f; --wait-bg:#332812;
  --dead:#e58079; --dead-bg:#34191a;
}}
:root[data-theme="dark"]{
  --ground:#12151a; --panel:#181c23; --line:#2b313b; --line-soft:#222831;
  --ink:#e6e9ef; --ink-2:#a2abbb; --ink-3:#6f7787;
  --accent:#8fb0e0;
  --run:#4fc98c; --run-bg:#14301f;
  --wait:#d7a63f; --wait-bg:#332812;
  --dead:#e58079; --dead-bg:#34191a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
     font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 64px;display:flex;flex-direction:column;gap:22px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;
       border-bottom:1px solid var(--line);padding-bottom:14px}
h1{margin:0;font-size:19px;font-weight:620;letter-spacing:-.01em}
.sub{color:var(--ink-3);font-size:12px;font-family:var(--mono);margin-top:3px}
.controls{display:flex;align-items:center;gap:8px}
button{font:inherit;font-size:12.5px;padding:6px 13px;border-radius:6px;cursor:pointer;
       border:1px solid var(--line);background:var(--panel);color:var(--ink)}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button[aria-pressed="true"]{background:var(--accent);color:var(--panel);border-color:var(--accent)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .n{font-family:var(--mono);font-size:25px;font-variant-numeric:tabular-nums;line-height:1.1}
.tile .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);margin-top:5px}
.tile.run .n{color:var(--run)} .tile.wait .n{color:var(--wait)}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
h2{margin:0;padding:11px 14px;font-size:12px;text-transform:uppercase;letter-spacing:.07em;
   color:var(--ink-2);border-bottom:1px solid var(--line-soft);font-weight:600}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:7px 14px;border-bottom:1px solid var(--line-soft);white-space:nowrap}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3);font-weight:600}
tr:last-child td{border-bottom:0}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink-2)}
td.script{font-family:var(--mono);color:var(--ink)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600;
      font-family:var(--mono)}
.pill.Running{background:var(--run-bg);color:var(--run)}
.pill.Waiting{background:var(--wait-bg);color:var(--wait)}
.pill.Error,.pill.Terminated{background:var(--dead-bg);color:var(--dead)}
.kind{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3)}
.empty{padding:16px 14px;color:var(--ink-3);font-size:12.5px}
.note{color:var(--ink-3);font-size:11.5px;font-family:var(--mono)}
</style>
<div class="wrap">
<header>
  <div>
    <h1>Grid'5000 queue</h1>
    <div class="sub" id="stamp"></div>
  </div>
  <div class="controls">
    <button id="refresh" type="button">Refresh</button>
    <button id="auto" type="button" aria-pressed="false">Auto</button>
    <button id="theme" type="button">Theme</button>
  </div>
</header>
<div class="tiles" id="tiles"></div>
<section><h2>GPU jobs &mdash; influence and eval arms</h2><div class="scroll"><div id="gpu"></div></div></section>
<section><h2>Generation &mdash; procedural data and collect</h2><div class="scroll"><div id="gen"></div></div></section>
<section><h2>Result cells, most recent</h2><div class="scroll"><div id="cells"></div></div></section>
<p class="note" id="src"></p>
</div>
<script>
const EMBEDDED = __DATA__;
let D = EMBEDDED, live = false;

const el = s => document.getElementById(s);
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

// Times come from the snapshot as epoch seconds, so the page reads the same on any machine.
const stamp = ts => ts ? new Date(ts*1000).toLocaleString([], {month:"short", day:"numeric",
                          hour:"2-digit", minute:"2-digit"}) : "\u2014";
const dur = h => h == null ? "\u2014" : h < 1 ? `${Math.round(h*60)}m`
                                       : h < 48 ? `${h.toFixed(1)}h` : `${(h/24).toFixed(1)}d`;
// A Running job reports how long it has run; anything else, how long it has been queued.
const age = r => r.state === "Running" ? dur(r.started_h) : dur(r.submitted_h);

function table(rows, cols){
  if(!rows.length) return '<p class="empty">Nothing queued.</p>';
  return '<table><thead><tr>' + cols.map(c=>`<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c=>`<td class="${c[2]||''}">${c[1](r)}</td>`).join('') + '</tr>').join('') +
    '</tbody></table>';
}

function render(){
  el("stamp").textContent = `snapshot ${D.generated} · ${D.jobs.length} jobs · ${D.cells_total} cells`;
  const st = D.by_state || {};
  el("tiles").innerHTML =
    `<div class="tile run"><div class="n">${st.Running||0}</div><div class="k">running</div></div>` +
    `<div class="tile wait"><div class="n">${st.Waiting||0}</div><div class="k">waiting</div></div>` +
    `<div class="tile"><div class="n">${(D.by_kind||{}).gpu||0}</div><div class="k">gpu jobs</div></div>` +
    `<div class="tile"><div class="n">${(D.by_kind||{}).gen||0}</div><div class="k">generation</div></div>` +
    `<div class="tile"><div class="n">${D.cells_total||0}</div><div class="k">result cells</div></div>`;

  const cols = [
    ["job", r=>`<span class="num">${esc(r.id)}</span>`, "num"],
    ["state", r=>`<span class="pill ${esc(r.state)}">${esc(r.state)}</span>`],
    ["site", r=>esc(r.site)],
    ["node", r=>esc(r.node||"—")],
    ["script", r=>esc(r.script||r.name||"—"), "script"],
    ["queue", r=>`<span class="kind">${esc(r.queue||"")}</span>`],
    ["submitted", r=>stamp(r.submitted), "num"],
    ["waiting/running", r=>age(r), "num"],
  ];
  const by = k => D.jobs.filter(j=>j.kind===k)
      .sort((a,b)=> (a.state===b.state ? a.id.localeCompare(b.id) : a.state==="Running" ? -1 : 1));
  el("gpu").innerHTML = table(by("gpu"), cols);
  el("gen").innerHTML = table(by("gen").concat(D.jobs.filter(j=>j.kind==="other")), cols);
  el("cells").innerHTML = table((D.recent_cells||[]).map(c=>({...c})), [
    ["cell", r=>esc(r.name), "script"],
    ["age", r=>`${r.age_h} h`, "num"],
  ]);
  el("src").textContent = live
    ? "Live · refreshed from status.json"
    : "Embedded snapshot. Refresh needs status.json served alongside this page (Tailscale); the artifact sandbox blocks it.";
}

async function refresh(){
  const b = el("refresh"); b.disabled = true; b.textContent = "…";
  try{
    const r = await fetch("status.json", {cache:"no-store"});
    if(!r.ok) throw new Error(r.status);
    D = await r.json(); live = true;
  }catch(e){ live = false; }
  b.disabled = false; b.textContent = "Refresh"; render();
}
el("refresh").addEventListener("click", refresh);

// Auto-refresh is 5 min, not 60 s, and stops entirely while the tab is hidden. A dashboard left
// open overnight should not keep a poller busy for a queue nobody is watching; it catches up on the
// next glance instead.
const PERIOD = 300000;
let timer = null;
function stop(){ if(timer){ clearInterval(timer); timer = null; } }
function start(){ stop(); if(!document.hidden) timer = setInterval(refresh, PERIOD); }
function autoOn(){ return el("auto").getAttribute("aria-pressed") === "true"; }
el("auto").addEventListener("click", ()=>{
  const on = autoOn();
  el("auto").setAttribute("aria-pressed", String(!on));
  el("auto").textContent = on ? "Auto" : "Auto · 5m";
  if(on) stop(); else { start(); refresh(); }
});
document.addEventListener("visibilitychange", ()=>{
  if(!autoOn()) return;
  if(document.hidden) stop(); else { start(); refresh(); }
});
el("theme").addEventListener("click", ()=>{
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
});
render();
refresh();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--status", default=str(here / "build" / "status.json"))
    ap.add_argument("--out", default=str(here / "build" / "g5k.html"))
    a = ap.parse_args()
    data = json.loads(Path(a.status).read_text())
    Path(a.out).write_text(HTML.replace("__DATA__", json.dumps(data)))
    print(f"[dashboard] {len(data.get('jobs', []))} jobs -> {a.out}")


if __name__ == "__main__":
    main()
