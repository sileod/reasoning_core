"""Why a wave failed, explained by a cheap model instead of by reading it yourself.

`trajectory` extracts what happened mechanically: gates, denials, where the steps went.
It cannot say *why*, and the why is buried in megabytes of tool output that nobody --
person or agent -- should pay to read. This sends the interesting trials to a small model
in one batched call and gets back a cause and a fix per trial, plus the patterns that
recur across them. The point is that the raw trajectory never enters the reader's context.

    python -m reasoning_core.task_search.digest runs/<arm>/<wave>/<stamp>

Interesting means: it did not succeed, or it succeeded only by exhausting its step budget.
Everything else is working and needs no explanation.
"""
import argparse
import json
import os
from pathlib import Path

from .trajectory import read, totals
from .wave_proposer import ChatClient

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_ENDPOINT = "https://albert.api.etalab.gouv.fr/v1/chat/completions"
DEFAULT_API_KEY_ENV = "ALBERT_API_KEY"
# The failure is at the end of a trajectory, never at the start, so transcripts are kept
# from the tail. These budgets put a 20-trial digest at roughly 25k tokens of input --
# one call, cheap, and small enough that a flash model still reads all of it.
TAIL_CALLS = 14
OUTPUT_CHARS = 240
TRIAL_CHARS = 3500

SYSTEM = """You debug autonomous coding agents. You are given trajectory excerpts from
agents asked to author one reasoning task each, and the harness verdict that judged them.
Diagnose the proximate cause of each outcome from the evidence shown. Distinguish an agent
mistake from a harness or prompt defect: if the agent was never told something it needed,
or a gate rejected work the instructions asked for, that is a pipeline bug and is the more
valuable finding. Never invent evidence you were not shown. Output one JSON object, no prose."""


def _tail(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ...[cut]"


def _transcript(trial_dir, row):
    """One trial, compressed to the parts that carry a diagnosis."""
    lines = [f"### {row['id']}  status={row['status']}"
             f"  steps={row['steps']}/{row['budget'] or '?'}  stop={row['stopped']}"]
    if row["checks"]:
        last = row["checks"][-1]
        failing = [gate for gate, verdict in last.items() if verdict == "FAIL"]
        lines.append("self-check (last of %d): %s"
                     % (len(row["checks"]), " ".join(failing) if failing else "all PASS"))
    else:
        lines.append("self-check: never run")
    for command in list(dict.fromkeys(row["denied"]))[:3]:
        lines.append("SANDBOX DENIED: " + _tail(command, 120))
    for name in row["errors"]:
        lines.append("API ERROR: " + str(name))
    for verdict in ("validation.log", "contract_audit.log"):
        path = trial_dir / verdict
        if path.is_file() and path.stat().st_size:
            lines.append(f"{verdict}: " + _tail(path.read_text(errors="replace")[-1200:], 600))
    lines.append("last tool calls:")
    for tool, ok, command, _size in row["calls"][-TAIL_CALLS:]:
        head = _tail(command, 120) or tool
        lines.append(f"  [{'ok ' if ok else 'ERR'}] {head}")
    if row["summary"]:
        lines.append("agent's closing line: " + _tail(row["summary"], 200))
    return _tail_block("\n".join(lines))


def _tail_block(text):
    return text if len(text) <= TRIAL_CHARS else text[-TRIAL_CHARS:]


def interesting(rows):
    """Successes that used their whole budget are near-misses, and worth explaining too."""
    return [row for row in rows
            if row["status"] != "success"
            or (row["budget"] and row["steps"] >= row["budget"] - 1)]


def _prompt(transcripts):
    return f"""Diagnose these {len(transcripts)} trials in exactly this JSON shape:
{{"trials": [{{"id": "T001", "cause": "one sentence", "blame": "agent|harness|task",
 "fix": "one concrete change, or none"}}],
 "patterns": [{{"pattern": "what recurs across trials", "trials": ["T001"],
 "fix": "the single change that would address all of them"}}]}}

Report one entry per trial, in the order given. Patterns are the payload: name at most
four, ordered by how many trials they explain, and only ones the evidence supports.

""" + "\n\n".join(transcripts)


def render(result, rows):
    lines = []
    causes = {item.get("id"): item for item in result.get("trials", [])}
    for row in rows:
        item = causes.get(row["id"], {})
        lines.append(f"{row['id']:<6} {row['status']:<24} [{item.get('blame', '?')}]"
                     f" {item.get('cause', 'not diagnosed')}")
        if item.get("fix") and item["fix"].lower() != "none":
            lines.append(f"       fix: {item['fix']}")
    lines.append("")
    for item in result.get("patterns", []):
        members = item.get("trials", [])
        lines.append(f"PATTERN ({len(members)}) {item.get('pattern', '')}")
        lines.append(f"        trials: {' '.join(members)}")
        lines.append(f"        fix:    {item.get('fix', '')}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wave", type=Path, help="a runs/<arm>/<wave>/<stamp> directory")
    parser.add_argument("--all", action="store_true",
                        help="digest every trial, not just the ones that need explaining")
    parser.add_argument("--limit", type=int, default=20,
                        help="most recent N interesting trials to send in the one call")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    directories = {directory.name: directory for directory in sorted(args.wave.iterdir())
                   if directory.is_dir() and (directory / "events.jsonl").is_file()}
    rows = [read(directory) for directory in directories.values()]
    if not rows:
        raise SystemExit(f"no trials with events.jsonl under {args.wave}")
    print(totals(rows))
    print()
    selected = rows if args.all else interesting(rows)
    if not selected:
        print("nothing to diagnose: every trial succeeded inside its budget")
        return 0
    selected = selected[-args.limit:]
    client = ChatClient(model=args.model, endpoint=args.endpoint,
                        api_key=os.environ.get(args.api_key_env),
                        temperature=0.2, reasoning_effort=None)
    transcripts = [_transcript(directories[row["id"]], row) for row in selected]
    result = client.json("digest", SYSTEM, _prompt(transcripts))
    print(json.dumps(result, indent=2) if args.json else render(result, selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
