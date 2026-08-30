"""What the worker actually did, read back out of events.jsonl.

A gate label names the artifact that was missing, never the reason it was missing.
The reason is in the trajectory: whether the self-check was run at all, what it
reported, which calls the sandbox denied, and whether the worker stopped because it
believed it was finished or because it ran out of steps. Reading eighteen of these
is how the largest measured improvement to this system was found -- the workers
were not running the self-check, because the prompt of that wave never named it.

    python -m reasoning_core.task_search.trajectory runs/ts_albert/WAVE0/<stamp>
"""
import argparse
import collections
import json
import re
from pathlib import Path

GATE = re.compile(r"^\s*(\w+)\s+(PASS|FAIL)\b", re.M)
DENIED = "prevents you from using"


def read(trial_dir):
    """One trial: its gates, its denials, and where its steps went."""
    events = [json.loads(line) for line in (trial_dir / "events.jsonl").open()] \
        if (trial_dir / "events.jsonl").is_file() else []
    run = json.loads((trial_dir / "run.json").read_text()) \
        if (trial_dir / "run.json").is_file() else {}
    summary = json.loads((trial_dir.parent / "summary.json").read_text()) \
        if (trial_dir.parent / "summary.json").is_file() else {}
    calls, denied, checks, texts, agy_steps = [], [], [], [], 0
    agy_stopped, agy_errors = None, []
    for event in events:
        if "event" in event:
            update = event.get("step_update", {})
            if (event.get("event") == "step_update"
                    and update.get("step_type") == "agent_response"):
                if update.get("text_delta"):
                    texts.append(update["text_delta"])
                if update.get("state") == "DONE":
                    agy_steps += 1
            if (event.get("event") == "step_update"
                    and update.get("step_type") == "tool"
                    and update.get("state") in {"DONE", "ERROR"}):
                info = update.get("tool_info", {})
                arguments = info.get("parameters", {})
                command = arguments.get("CommandLine", "")
                ok = update.get("state") == "DONE"
                output = str(info.get("output") or info.get("error") or "")
                calls.append((update.get("tool_name", "tool"), ok, command,
                              len(output)))
                if (not ok and re.search(
                        r"permission|denied|not allowed", output, re.I)):
                    denied.append(command or update.get("tool_name", "tool"))
                if ok and "task_search.selfcheck" in command:
                    gates = dict(GATE.findall(output))
                    if not re.search(r"^\d+ gate\(s\) failing\.\s*$", output, re.M):
                        gates["incomplete"] = "FAIL"
                    checks.append(gates)
            if event.get("event") == "result":
                agy_stopped = event.get("result", {}).get("status")
                if agy_stopped not in {None, "SUCCESS"}:
                    agy_errors.append(agy_stopped)
            continue
        if event["type"] == "text":
            texts.append(event["part"]["text"])
        if event["type"] != "tool_use":
            continue
        state = event["part"]["state"]
        command = (state.get("input") or {}).get("command", "")
        ok = state.get("status") == "completed"
        calls.append((event["part"]["tool"], ok, command, len(str(state.get("output") or ""))))
        if not ok and DENIED in str(state.get("error", "")):
            denied.append(command or event["part"]["tool"])
        if ok and "task_search.selfcheck" in command:
            output = str(state.get("output") or "")
            gates = dict(GATE.findall(output))
            # A harness timeout can leave a completed tool event containing only the
            # early PASS lines. S35 looked all-green live even though its self-check
            # had been killed before samples, pytest, contract and gameability ran.
            if not re.search(r"^\d+ gate\(s\) failing\.\s*$", output, re.M):
                gates["incomplete"] = "FAIL"
            checks.append(gates)
    return {
        "id": trial_dir.name,
        "status": run.get("status", "no run.json"),
        "steps": (agy_steps if agy_steps else
                  sum(1 for e in events if e.get("type") == "step_start")),
        # run.json appears only after every coordinator gate completes. During the
        # model phase the invocation summary is the durable source of this value.
        "budget": ((run.get("generation") or {}).get("settings", {}).get("max_steps")
                   or summary.get("max_steps")),
        "calls": calls,
        "denied": denied,
        "checks": checks,
        "errors": (agy_errors or [e["error"]["name"] for e in events
                                   if e.get("type") == "error"]),
        "stopped": (agy_stopped if agy_stopped is not None else
                    events[-1]["part"].get("reason") if events and
                    events[-1].get("type") == "step_finish" else None),
        "summary": texts[-1].strip().splitlines()[0][:120] if texts else "",
    }


def report(rows):
    lines = []
    for row in rows:
        budget = row["budget"] or 0
        wall = " AT-BUDGET" if budget and row["steps"] >= budget - 1 else ""
        lines.append(
            f"{row['id']:<5} {row['status']:<24} steps={row['steps']:>3}/{budget or '?':<3}"
            f" denied={len(row['denied']):>2} selfcheck={len(row['checks']):>2}"
            f" stop={row['stopped']}{wall}")
        # The last self-check is the one whose verdict the worker acted on, or failed to.
        if row["checks"]:
            failing = [g for g, v in row["checks"][-1].items() if v == "FAIL"]
            lines.append(f"      last self-check: {'FAIL ' + ' '.join(failing) if failing else 'all PASS'}")
        else:
            lines.append("      never ran the self-check")
        for command in list(dict.fromkeys(row["denied"]))[:3]:
            lines.append(f"      DENIED  {command.splitlines()[0][:96]}")
        for name in row["errors"]:
            lines.append(f"      API ERROR  {name}")
    return "\n".join(lines)


def totals(rows):
    status = collections.Counter(r["status"] for r in rows)
    calls = sum(len(r["calls"]) for r in rows)
    denied = sum(len(r["denied"]) for r in rows)
    context = sum(size for r in rows for _, _, _, size in r["calls"])
    at_budget = [r["id"] for r in rows if r["budget"] and r["steps"] >= r["budget"] - 1]
    silent = [r["id"] for r in rows if not r["checks"]]
    return "\n".join((
        f"trials {len(rows)}  " + "  ".join(f"{k}={v}" for k, v in status.most_common()),
        f"success {sum(1 for r in rows if r['status'] == 'success')}/{len(rows)}"
        f"   scratch {_rate(rows, ('N', 'S'))}   mutation {_rate(rows, 'M')}",
        f"calls {calls}, denied {denied} ({100 * denied / max(calls, 1):.1f}%),"
        f" tool output {context // 1024}KB back into context",
        f"ran out of steps: {' '.join(at_budget) or 'none'}",
        f"never self-checked: {' '.join(silent) or 'none'}",
    ))


def _rate(rows, prefix):
    group = [r for r in rows if r["id"].startswith(prefix)]
    if not group:
        return "n/a"
    return f"{sum(1 for r in group if r['status'] == 'success')}/{len(group)}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wave", type=Path, help="a runs/<arm>/WAVE0/<stamp> directory")
    parser.add_argument("--json", action="store_true", help="emit the rows instead")
    args = parser.parse_args(argv)
    rows = [read(d) for d in sorted(args.wave.iterdir())
            if d.is_dir() and (d / "events.jsonl").is_file()]
    if args.json:
        for row in rows:
            row.pop("calls")
        print(json.dumps(rows, indent=2))
        return 0
    print(report(rows))
    print()
    print(totals(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
