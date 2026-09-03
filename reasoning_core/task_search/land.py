"""Copy a finished wave's chosen drafts out of their worktrees and into the package.

Landing is not promotion. A landed task is importable and addressable by name -- which is
what the influence pipeline and the zero-shot probes need -- but `list_tasks()` still
leaves it out, because nothing has yet shown it is worth a place on the roster. Promotion
is what happens after influence has an opinion.

Triage decides *which* draft: one idea can have several, and `pick` orders them. This
copies that pick's owned directory, refuses anything triage marked drop or gameable, and
then checks the only thing a copy can get wrong -- that the task really does load, draw
and score under the package it just joined.

    python -m reasoning_core.task_search.land runs/ts_wave8/wave8 \
        --plan reasoning_core/task_search/plans/wave8.yaml
    python -m reasoning_core.task_search.land runs/ts_wave8/wave8 --plan ... --apply

Without --apply it reports what would land and copies nothing.
"""
import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .. import _discover_tasks, _task_to_module_map, prepr_task_name
from .plan import load_plan
from .triage import _mark, _recorded_verdict, draft, pick, proposal_of, successes


# Model-written code lives under generated/ and nowhere else. The target comes from the
# plan, and a plan is a file: this is the one place that can hold the line, so it holds it
# regardless of what the plan asks for.
GENERATED_ROOT = Path("reasoning_core/tasks/generated")


def outside_generated(target):
    """True if the plan is asking to write model-written code into the package proper."""
    try:
        Path(target).resolve().relative_to(GENERATED_ROOT.resolve())
    except ValueError:
        return True
    return False


# A task that lands has to survive the round trip through the package: discovered by name,
# drawn at the bottom and the top of its ladder, and scoring its own answer. Run out of
# process, because discovery is computed once at import and this runs right after a copy.
CHECK = """
import sys
import reasoning_core as rc
name, module = sys.argv[1], sys.argv[2]
if name not in rc.list_tasks(include_generated=True):
    print("not discovered"); raise SystemExit(1)
found = rc._task_to_module_map[name][0]
if found != module and not found.startswith(module + "."):
    print(f"the name is already taken by {found}"); raise SystemExit(1)
task = rc.get_task(name)
for level in (0, 6):
    example = task.generate_example(level=level)
    if not example.prompt:
        print(f"empty prompt at level {level}"); raise SystemExit(1)
    if task.score_answer(example.answer, example) != 1:
        print(f"own answer does not score 1 at level {level}"); raise SystemExit(1)
"""


# A wave draws each idea more than once and names the draws apart -- `_v1`, `_v2` -- so
# two workers can build the same idea without colliding, and an implementor will sometimes
# add `Task` to the class name on top of that. None of it is part of the task's identity.
VARIANT = re.compile(r"_v\d+$")
NOISE = re.compile(r"(_v\d+|_task)$")
CLASS_VARIANT = re.compile(r"\b([A-Za-z_]\w*?)V\d+\b")
TEXT = (".py", ".md", ".json", ".yaml", ".txt")


def unsuffixed(name):
    return VARIANT.sub("", name)


def strip_variant(target, name):
    """Rewrite the draft's variant suffix out of the copied files and their filenames."""
    base = unsuffixed(name)
    if base == name:
        return

    def declassify(match):
        return match.group(1) if prepr_task_name(match.group(0)) == name else match.group(0)

    for path in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.suffix in TEXT:
            body = path.read_text()
            rewritten = CLASS_VARIANT.sub(declassify, body.replace(name, base))
            if rewritten != body:
                path.write_text(rewritten)
        if name in path.name:
            path.rename(path.with_name(path.name.replace(name, base)))


# What the package already answered to before this run copied anything. Two waves do
# occasionally build the same idea, and discovery resolves a repeated name by silently
# letting one module win -- so the second one to arrive is refused here instead.
CLAIMED = {name: module for name, (module, _) in _task_to_module_map.items()}


def settle_name(target):
    """(name, problem) -- what the copy will answer to, which is what it declares minus the
    noise a trial leaves on it. The plan's directory name does not get a vote: older waves
    prefix theirs with a proposal id, and that is bookkeeping, not a task name. Renaming
    the class is not safe either -- drafts define a helper `BridgeEdges` beside their
    `BridgeEdgesTask` -- so a rename goes through `task_name`, the seam discovery has for
    exactly this.
    """
    found = _discover_tasks(target)[0]
    if len(found) != 1:
        return None, f"defines {len(found)} tasks, not one"
    [(declared, (module, class_name))] = found.items()
    name = NOISE.sub("", declared)
    if name != declared:
        path = target / f"{module.replace('.', '/')}.py"
        lines = path.read_text().splitlines(keepends=True)
        node = next(n for n in ast.walk(ast.parse("".join(lines)))
                    if isinstance(n, ast.ClassDef) and n.name == class_name)
        first = node.body[0]
        docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
        lines.insert(first.end_lineno if docstring else first.lineno - 1,
                     f'{" " * first.col_offset}task_name = "{name}"\n')
        path.write_text("".join(lines))
    return name, None


def owned_dir(row, plan_trial):
    """The worktree directory holding the draft, as the plan assigned it."""
    return Path(row["dir"]) / "worktree" / plan_trial.owned_path


def check(name, module, timeout=300):
    """None if the landed task loads, draws and scores, under its own name; else why not."""
    done = subprocess.run([sys.executable, "-c", CHECK, name, module],
                          capture_output=True, text=True, timeout=timeout)
    if done.returncode == 0:
        return None
    said = done.stdout.strip() or done.stderr.strip()
    return said.splitlines()[-1] if said else "failed"


def landable(wave_root, plan_path):
    """(mark, row, plan_trial) per idea, best draft first, in idea order."""
    plan_trials = {trial.trial_id: trial for trial in load_plan(plan_path).trials}
    groups = {}
    for trial_dir, trial in successes(wave_root):
        verdict, source = _recorded_verdict(trial_dir, trial)
        groups.setdefault(proposal_of(trial["trial_id"]), []).append(
            (draft(trial_dir, trial, verdict, source), trial))
    chosen = []
    for proposal in sorted(groups):
        rows = [row for row, _ in groups[proposal]]
        best = pick(rows)[0]
        chosen.append((_mark(best), best, plan_trials.get(best["trial"])))
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wave", type=Path, help="a runs/<arm>/<wave> directory")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true",
                        help="copy the drafts in; without it nothing is written")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a task directory that is already in the package")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    landed, skipped = [], []
    for mark, row, plan_trial in landable(args.wave, args.plan):
        name, source = row["name"], None
        if mark != "take" or plan_trial is None:
            skipped.append({"name": name, "why": mark if plan_trial else "not in plan"})
            continue
        source = owned_dir(row, plan_trial)
        # The variant suffix is a trial artifact; the task lands under the bare idea name.
        target = Path(plan_trial.owned_path)
        target, name = target.with_name(unsuffixed(target.name)), unsuffixed(name)
        if outside_generated(target):
            skipped.append({"name": name, "why": f"target outside {GENERATED_ROOT}"})
            continue
        if not source.is_dir():
            skipped.append({"name": name, "why": "worktree is gone"})
            continue
        if target.exists() and not args.overwrite:
            skipped.append({"name": name, "why": "already in the package"})
            continue
        if not args.apply:
            landed.append({"name": name, "trial": row["trial"], "target": str(target)})
            continue
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        strip_variant(target, row["name"])
        name, problem = settle_name(target)
        claim = CLAIMED.get(name)
        module = ".".join((*target.parts[2:-1], name))
        if claim and claim != module and not claim.startswith(module + "."):
            problem = f"the name is already taken by {claim}"
        if not problem and target.name != name:
            if target.with_name(name).exists() and not args.overwrite:
                problem = "already in the package"
            else:
                shutil.rmtree(target.with_name(name), ignore_errors=True)
                target = target.rename(target.with_name(name))
        problem = problem or check(name, module)
        if problem:
            # A task that cannot load is worse than a missing one: it breaks discovery for
            # everything that lands after it, so it goes straight back out.
            shutil.rmtree(target)
            skipped.append({"name": name or row["name"], "why": problem})
            continue
        CLAIMED[name] = module
        landed.append({"name": name, "trial": row["trial"], "target": str(target)})

    for row in landed:
        print(f"{'landed' if args.apply else 'would land':<12} {row['name']:<46} {row['trial']}")
    for row in skipped:
        print(f"{'skipped':<12} {row['name']:<46} {row['why']}")
    print(f"\n{len(landed)} landed, {len(skipped)} skipped")
    if args.json_out:
        args.json_out.write_text(json.dumps({"landed": landed, "skipped": skipped}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
