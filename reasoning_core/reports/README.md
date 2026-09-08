# reports

Reusable report tooling. Everything here is tracked, unlike `task_diagnostics/` and `paper_exports/`,
so another agent gets it from a plain `git pull`.
`per_task_results/` is an ignored local input directory used by these reports;
a fresh clone does not contain measurements. Published tables live in
[`task_influence/`](../../task_influence/RESULTS.md).

| module | what it does |
|---|---|
| `protocol_calibration.py` | maps warm-protocol (T75/M80) effects onto the standard (T300/M20) axis. Per-leg OLS; legs with \|r\| < 0.5 pass through uncalibrated. |
| `atlas_data.py` | rebuilds the Atlas `const D` block from `per_task_results`, calibrating any warm source. **Merges** — it preserves `legs`/`pc`/`ax`, which come from a different pipeline. |
| `item_factors.py` | exploratory factor analysis over individual eval ITEMS from the `--per-example-dir` sidecars, varimax-rotated. |
| `task_staleness.py` | which task edits have never been measured: live `behavior_hash` vs the hash every manifest recorded. `--names` feeds the rebuild directly. |
| `refresh.py` | one command for "I edited some tasks": probes the drifted ones against the live generators, rebuilds the pages, writes the influence submission. Plans by default. |
| `g5k_status.py` | polls **every** G5K site (oarstat is site-local) and writes `build/status.json`. |
| `build_dashboard.py` | renders `build/g5k.html` from that snapshot. |
| `serve.sh` | serves `build/` on the tailnet address only. |

## Rebuild and publish the Atlas

```bash
python -m reasoning_core.reports.atlas_data --patch-html reasoning_core/reports/build/index.html
reasoning_core/reports/serve.sh 8778          # http://<tailscale-ip>:8778/
```

Do not hand-edit the `const D = {...}` block — the next rebuild overwrites it. Add a source to
`SOURCES` in `atlas_data.py` instead. Prose sections of the HTML are safe to edit by hand.

## Protocols do not mix

Warm and standard protocols agree on **ranking** (rho +0.905) but not on **level**: the warm protocol
applies a per-leg gain, median 1.38x, from 0.39 (`winogrande`) to 3.13 (`themis_py`). Raw warm rows
in a standard matrix read about half again too strong. `atlas_data.py` handles this; if you build any
other mixed-protocol figure, use `protocol_calibration.fit`.

The two anchors every run carries confirm the direction but cannot fit the map — anchors are
deliberately low-effect, so they sit near the origin where a slope is unidentifiable.

## Item-level factors

```bash
python -m reasoning_core.reports.item_factors \
  --per-example <dir> --cells 'influence_COLL-roster_W[LS]SH*_T75_M80_*.json' --k 4
```

Sidecars join to arms through the stored metric vector, not the filename hash. The cell carries the
`_delta` aliases too, so the join projects the cell onto the sidecar's key set.

## G5K queue dashboard

```bash
python -m reasoning_core.reports.g5k_status --out reasoning_core/reports/build/status.json
python -m reasoning_core.reports.build_dashboard
reasoning_core/reports/serve.sh 8778        # http://<tailscale>:8778/g5k.html
```

Served over Tailscale, the page's **Refresh** and **Auto 60s** buttons re-fetch `status.json` from
the same directory, so other agents get live state. Published as a Claude artifact the CSP blocks
that fetch, so the embedded snapshot is what shows and the footer says so rather than pretending it
is live.

`scratchpad/g5k_watch.sh` keeps `status.json` current on a 5-minute nohup loop. Not cron: cron is
enabled on this host but user jobs never fire.

Job kind is inferred from the script name in the OAR command, so `gpu` (influence/eval arms) and
`gen` (generator array + collect) stay in separate tables — mixed together, whichever is smaller
disappears.

**Tooling is public; measurements are not.** The modules here are tracked so another agent gets
them from a plain `git pull`. `build/` stays gitignored -- it holds result snapshots, queue
telemetry and pages with embedded data. Credentials are always read from the environment or a key
file (`$NVIDIA_NIM_API_KEY`, falling back to `~/.nvapi_key`); never inline one here.
