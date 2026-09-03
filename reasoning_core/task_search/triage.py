"""Which of a finished wave's successes are actually worth promoting.

A wave reports successes, not a shipping list. Two things stand between the two, and
neither is visible in `trajectory`'s status counter.

The first is that the semantic reviewer fails open. `_sample_sanity` returns a null
verdict when the endpoint is unreachable, and `semantics` reads `verdict != "INVALID"`,
so an outage passes every candidate it could not read. That is the right call for the
gate -- a reviewer outage must not reject work that is fine -- and the wrong one for
promotion, where "nobody checked" is not "checked and clean". wave8 ran into Albert's
rate limit and 108 of its 141 successes were never reviewed at all.

The second is that a wave draws each proposal more than once. wave8's 141 successes are
77 ideas, 48 of them built twice by two workers who never saw each other's answer.
Promoting both drafts puts near-duplicates in the catalog, and every later novelty check
then measures against them.

So: group the successes by proposal, run the review that never ran, and print one line
per idea saying which draft to take.

    python -m reasoning_core.task_search.triage runs/ts_wave8/wave8 --plan .../wave8.yaml
    python -m reasoning_core.task_search.triage runs/ts_wave8/wave8 --plan ... --review

Without --review it reports what is already known and reviews nothing. Reviews are cached
in the trial directory, so a re-run costs nothing for the trials already judged.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .plan import load_plan
from .validation import _review_source, _sample_sanity

CACHE_NAME = "sample_sanity.retry.json"
AUDIT_NAME = "prior_audit.n40.json"
# The in-trial gameability audit runs at n=30, which is noisy right at the 0.40 ceiling:
# wave0's n02 cleared it at n=30 and lost at n=40, which is how it reached promotion.
# Forty is the number that decided then, and at four seconds a task it is nearly free.
AUDIT_SAMPLES = 40
AUDIT_CEILING = 0.4
# The worker's audit stops sampling a level after 45 seconds, so a slow generator can be
# judged on whatever n it reached by then rather than on the n that was asked for.
# Triage is not on a clock. (The n prior_audit reports is smaller than this anyway: it is
# the balanced subset, after the shipping label cap discards the skew.)
AUDIT_BUDGET_SECONDS = 600
# The reviewer shares one per-minute token bucket with everything else pointed at the
# provider, and a triage pass is not urgent. One call every few seconds keeps it under
# the limit without the retry machinery having to get involved.
REVIEW_PAUSE_SECONDS = 4
DRAFT_SUFFIX = re.compile(r"v\d+$")


def proposal_of(trial_id):
    """P001v2 -> P001. Drafts of one idea share a proposal."""
    return DRAFT_SUFFIX.sub("", trial_id)


def successes(wave_root):
    """Every successful trial under a wave, newest run last, canonical attempts only."""
    found = []
    for run in sorted(Path(wave_root).iterdir()):
        if not run.is_dir():
            continue
        for record in sorted(run.glob("*/run.json")):
            # `P003v2.attempt1-provider_429` is a retry's corpse, not a second result.
            if "." in record.parent.name:
                continue
            trial = json.loads(record.read_text())
            if trial.get("status") == "success":
                found.append((record.parent, trial))
    return found


def _recorded_verdict(trial_dir, trial):
    """The verdict this trial has, from the run or from an earlier triage pass."""
    cached = trial_dir / CACHE_NAME
    if cached.is_file():
        return json.loads(cached.read_text()), "triage"
    return dict(trial.get("sample_sanity") or {"verdict": None, "why": ""}), "run"


def review(trial_dir, trial, plan_trial):
    """Run the semantic review this trial never got, and remember the answer."""
    worktree = Path(trial["worktree"])
    samples = sorted((worktree / plan_trial.owned_path).glob("samples_*.md"))
    if not samples:
        return {"verdict": None, "why": "no samples file survives in the worktree"}
    verdict = _sample_sanity(
        samples[0],
        instruction=plan_trial.instruction,
        source=_review_source(worktree, plan_trial.owned_path),
    )
    (trial_dir / CACHE_NAME).write_text(json.dumps(verdict, indent=1))
    return verdict


def audit(trial_dir, trial, plan_trial):
    """Re-run the gameability audit at the sample size that decides, not the one that fits.

    The wave runs this at n=30 inside a 45-second worker budget, where a task sitting
    near the ceiling passes or fails mostly by luck. Nothing here is on a clock.
    """
    cached = trial_dir / AUDIT_NAME
    if cached.is_file():
        return json.loads(cached.read_text())
    finished = subprocess.run(
        [sys.executable, "-m", "reasoning_core.task_search.prior_audit",
         "--path", plan_trial.owned_path, "--n", str(AUDIT_SAMPLES),
         "--max-const", str(AUDIT_CEILING), "--max-shortcut", str(AUDIT_CEILING),
         "--budget-seconds", str(AUDIT_BUDGET_SECONDS)],
        cwd=trial["worktree"], capture_output=True, text=True, timeout=900,
        # The audit needs no credentials, and the candidate source it imports is
        # untrusted, so it gets an environment with nothing to spend.
        env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
             "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(Path.home())},
    )
    tail = (finished.stdout or finished.stderr).strip().splitlines()
    result = {"ok": finished.returncode == 0, "why": tail[-1] if tail else "no output"}
    cached.write_text(json.dumps(result, indent=1))
    return result


def _task_name(trial):
    for path in trial.get("changed_paths") or []:
        parts = path.split("/")
        if len(parts) > 4:
            return parts[4]
    return trial["trial_id"]


def draft(trial_dir, trial, verdict, source, audited=None):
    steps = trial.get("steps") or {}
    return {
        "audit": audited,
        "trial": trial["trial_id"],
        "run": trial_dir.parent.name,
        "name": _task_name(trial),
        "dir": str(trial_dir),
        "verdict": verdict.get("verdict"),
        "why": verdict.get("why", ""),
        "reviewed_by": source,
        "steps": steps.get("used"),
        "budget": steps.get("max"),
        # A success that used its whole budget stopped because it ran out, not because
        # it was done, so it is the weaker draft when there is a choice.
        "exhausted": bool(steps.get("exhausted")),
    }


# A draft with no verdict is not equal to one that was read and cleared, and it is not
# equal to one that was read and rejected either. Rank on that, then on whether the
# worker finished with budget to spare.
RANK = {"VALID": 0, None: 1, "INVALID": 2}


def pick(drafts):
    # Newest run last, so a re-run of the same trial id wins the tie: it is the one whose
    # worktree the rest of the wave was measured against.
    return sorted(drafts, key=lambda d: (RANK.get(d["verdict"], 1), d["exhausted"],
                                         d["trial"], _newest(d)))


def _newest(row):
    """Later run first among otherwise equal drafts."""
    return tuple(-ord(character) for character in row.get("run", ""))


def render(groups):
    lines = []
    for proposal, drafts in sorted(groups.items()):
        best, *rest = pick(drafts)
        mark = _mark(best)
        lines.append(f"{proposal:<6} {mark:<10} {best['trial']:<8} {best['name']:<46}"
                     f" {best['steps']}/{best['budget']}")
        for other in rest:
            lines.append(f"{'':6} {'also':<10} {other['trial']:<8} {other['name']:<46}"
                         f" {other['steps']}/{other['budget']} [{other['verdict']}]")
        if best["verdict"] == "INVALID":
            lines.append(f"{'':17}why: {best['why']}")
        if best.get("audit") and not best["audit"]["ok"]:
            lines.append(f"{'':17}audit: {best['audit']['why']}")
    return "\n".join(lines)


def _mark(best):
    """A gameable task does not ship however cleanly the reviewer read it."""
    if best.get("audit") and not best["audit"]["ok"]:
        return "gameable"
    return {"VALID": "take", None: "unreviewed", "INVALID": "drop"}[best["verdict"]]


def summarize(groups):
    counts = {"take": 0, "unreviewed": 0, "drop": 0, "gameable": 0}
    for drafts in groups.values():
        counts[_mark(pick(drafts)[0])] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wave", type=Path, help="a runs/<arm>/<wave> directory")
    parser.add_argument("--plan", type=Path, required=True,
                        help="the plan the wave ran, for each trial's assignment")
    parser.add_argument("--review", action="store_true",
                        help="run the semantic review for trials that never got one")
    parser.add_argument("--audit", action="store_true",
                        help="re-run the gameability audit at n=%d on each idea's pick"
                             % AUDIT_SAMPLES)
    parser.add_argument("--limit", type=int, default=0,
                        help="review at most this many trials in one pass; 0 is all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    plan_trials = {trial.trial_id: trial for trial in load_plan(args.plan).trials}
    found = successes(args.wave)
    if not found:
        raise SystemExit(f"no successful trials under {args.wave}")

    groups, reviewed = {}, 0
    for trial_dir, trial in found:
        verdict, source = _recorded_verdict(trial_dir, trial)
        plan_trial = plan_trials.get(trial["trial_id"])
        if (args.review and verdict.get("verdict") is None and plan_trial
                and (not args.limit or reviewed < args.limit)):
            if reviewed:
                time.sleep(REVIEW_PAUSE_SECONDS)
            verdict, source, reviewed = review(trial_dir, trial, plan_trial), "triage", reviewed + 1
        groups.setdefault(proposal_of(trial["trial_id"]), []).append(
            (draft(trial_dir, trial, verdict, source), trial, plan_trial))

    for name, entries in groups.items():
        # Audit only the draft that would actually ship; the runner-up costs the same
        # four seconds and nobody is going to promote it.
        chosen = pick([entry[0] for entry in entries])[0]
        groups[name] = [entry[0] for entry in entries]
        if not args.audit:
            continue
        for row, trial, plan_trial in entries:
            if row is chosen and plan_trial:
                row["audit"] = audit(Path(row["dir"]), trial, plan_trial)

    if args.json:
        print(json.dumps({name: pick(drafts) for name, drafts in sorted(groups.items())},
                         indent=1))
        return 0
    print(render(groups))
    counts = summarize(groups)
    print(f"\n{len(found)} successes, {len(groups)} ideas: "
          + ", ".join(f"{value} {key}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
