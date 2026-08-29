"""Every gate the harness applies, in one command.

Trials were spending half a 28-step budget on five separate verification commands and
the edit/re-verify cycles between them, and the gates that are not in that list --
TASK_META, the contract audit, the sample headings -- only surfaced in run.json once
the trial was already lost. Reporting all of them in one call makes a debug cycle one
edit plus one check, and makes the silent gates visible while there is budget to fix
them.

    python -m reasoning_core.task_search.selfcheck <owned_path> <trial_id>

Each gate prints PASS, FAIL with the reason, or SKIP when a prerequisite failed.
Exit code is the number of failed gates.

The whole run has to fit in the 300 seconds the agent harness allows a single bash
call, so gates run cheapest-first behind a shared deadline, every line is flushed as
it is produced, and a gate that would overrun is skipped rather than allowed to eat
the budget and have the entire report killed with nothing printed.
"""
import argparse, ast, hashlib, json, os, pathlib, shlex, subprocess, sys, time

# Kept in step with runner.SAMPLE_SECTIONS by a test, not by an import: this
# module runs inside the sandbox, where importing the runner would drag in yaml
# and the package, and a self-check that cannot start reports nothing at all.
SECTIONS = ("level 0", "level 2", "level 5", "answer")
# opencode kills a bash call at 300s (runner.py's _mini_config) and the harness gives
# each validation command the same. Stop early enough to print the summary.
DEADLINE = time.monotonic() + 240
CONTRACT_EXAMPLES = 64
_PROBE_N = 8
_PROBE = r"""
import importlib, json, random, sys, time
classes, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
costs = []
for offset, (module_name, class_name) in enumerate(classes):
    task = getattr(importlib.import_module(module_name), class_name)()
    random.seed(seed + offset)
    for _ in range(n):
        started = time.monotonic()
        task.generate_example()
        costs.append(time.monotonic() - started)
print(sum(costs) / len(costs), max(costs))
"""


def remaining():
    return DEADLINE - time.monotonic()


def contract_audit_source():
    """runner.py's contract audit, read from runner.py so the two cannot drift."""
    tree = ast.parse(pathlib.Path(__file__).with_name("runner.py").read_text())
    return ast.literal_eval(next(
        n.value for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_CONTRACT_AUDIT"))


def sh(command, env=None, limit=None):
    """Run one shell command the way _run_validation does, from the worktree root.

    Bounded by whatever is left of the shared deadline, so no single slow generator can
    consume the budget and leave the report unprinted.
    """
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {}))
    budget = min(limit or remaining(), remaining())
    if budget <= 1:
        return 124, "out of time before this command started"
    try:
        done = subprocess.run(["/bin/bash", "-c", command], capture_output=True,
                              text=True, env=environment, timeout=budget)
    except subprocess.TimeoutExpired:
        return 124, "killed after %d s" % budget
    return done.returncode, (done.stdout + done.stderr)


def tail(text, n=12):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join("    | " + line for line in lines[-n:])


class Report:
    def __init__(self):
        self.failed, self.stop = 0, False

    def gate(self, name, ok, detail="", fatal=False):
        if ok is None:
            print(f"{name:<14} SKIP  {detail}", flush=True)
            return False
        print(f"{name:<14} {'PASS' if ok else 'FAIL'}  {detail}".rstrip(), flush=True)
        if not ok:
            self.failed += 1
            self.stop = self.stop or fatal
        return ok


def module_facts(root):
    """Task subclasses and TASK_META assignments under root, read the way runner.py reads them."""
    from reasoning_core.task_search.runner import _task_classes, _task_metadata
    worktree = pathlib.Path.cwd()
    return _task_classes(worktree, root), _task_metadata(worktree, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("owned_path")
    parser.add_argument("trial_id")
    parser.add_argument("--n", type=int, default=30, help="gameability sample count")
    args = parser.parse_args(argv)
    owned, trial = args.owned_path.rstrip("/"), args.trial_id
    generator = f"{owned}/generate_samples_{trial}.py"
    samples = pathlib.Path(owned) / f"samples_{trial}.md"
    report = Report()

    if not pathlib.Path(owned).is_dir():
        report.gate("implementation", False, f"{owned} does not exist")
        return 1
    try:
        classes, metas = module_facts(owned)
    except SyntaxError as error:
        report.gate("implementation", False, f"{error.filename}:{error.lineno} {error.msg}")
        return 1
    ok = report.gate("implementation", bool(classes),
                     ", ".join(f"{m}.{c}" for m, c in classes) or
                     "no class subclassing Task under the owned path", fatal=True)

    # Passing implementation is not the same as shipping: _discover_tasks walks the
    # tasks tree by AST and drops anything under a directory or file whose name starts
    # with "_" or ".", so a task can validate perfectly and still never reach DATASETS.
    from reasoning_core import _discover_tasks
    shipped, _ = _discover_tasks("reasoning_core/tasks")
    mine = sorted(n for n, (module, _) in shipped.items()
                  if module.replace(".", "/") in
                  {m.split("reasoning_core.tasks.", 1)[-1].replace(".", "/") for m, _ in classes})
    report.gate("discovery", bool(mine),
                ", ".join(mine) or "no class under the owned path reaches DATASETS;"
                " check that no directory or file on the path starts with _ or .")

    spec = os.environ.get("TASK_SEARCH_SPEC")
    want = json.load(open(spec))["task_meta"] if spec and os.path.exists(spec) else None
    if len(metas) != 1:
        report.gate("task_meta", False,
                    f"found {len(metas)} module-level TASK_META assignments, need exactly 1"
                    + (": " + ", ".join(p for p, _ in metas) if metas else ""))
    elif want is not None and metas[0][1] != want:
        keys = sorted(set(want) | set(metas[0][1]))
        bad = [k for k in keys if want.get(k) != metas[0][1].get(k)]
        report.gate("task_meta", False, f"differs from the required mapping at {bad}")
    else:
        report.gate("task_meta", True, metas[0][0] +
                    ("" if want is not None else " (presence only; required value unknown)"))
    if report.stop:
        return report.failed

    started = time.monotonic()
    code, out = sh(
        "python -c \"import importlib,random,sys,json;"
        "cs=json.loads(sys.argv[1]);"
        "[[t.config.set_level(L) or t.validate(n_samples=3) for L in (0,2,5)]"
        " for t in [getattr(importlib.import_module(m),c)() for m,c in cs]];"
        f"print('SMOKE_OK')\" '{json.dumps(classes)}'", limit=120)
    elapsed = time.monotonic() - started
    if not report.gate("smoke", code == 0, "levels 0, 2, 5 validate in %.0fs" % elapsed
                       if code == 0 else "validate() failed\n" + tail(out), fatal=True):
        return report.failed

    # The harness allows a validation command 300 seconds and its contract audit has to
    # generate 64 examples at the default config, so a generator averaging more than
    # ~4.5s an example loses the trial on a clock nothing reports -- M10 in wave
    # 20260829T072855Z did, with an exit-124 run.json it never got to see. Time eight
    # examples the way the contract audit makes them, and quote the worst as well as the
    # mean: the cost is heavy-tailed, so an average read off cheap instances is not a
    # prediction. M10 averaged 7.1s over eight with one at 33s.
    code, out = sh("python -c %s %s %d %d" % (
        shlex.quote(_PROBE), shlex.quote(json.dumps(classes)),
        int(hashlib.sha256(trial.encode()).hexdigest()[:6], 16), _PROBE_N),
        limit=min(90, remaining()))
    if code != 0:
        report.gate("speed", False, "eight examples did not finish in 90s, so the 64 the"
                    " contract audit generates have no chance of finishing in 300")
    else:
        mean, worst = (float(x) for x in out.split()[-2:])
        projected = mean * CONTRACT_EXAMPLES
        report.gate("speed", projected < 240,
                    "%.2fs an example on average, worst %.2fs, so %s for the %d the"
                    " contract audit generates and the harness kills it at 300"
                    % (mean, worst, "%.0fs" % projected if projected >= 1
                       else "well under a second", CONTRACT_EXAMPLES)
                    + ("" if mean * CONTRACT_EXAMPLES < 240 else
                       ". Make generate_example cheaper at the DEFAULT config, which is"
                       " what the audit uses: bound the retries in a rejection-sampling"
                       " loop, shrink the search it runs, or cache what does not depend"
                       " on the instance. A task this slow fails the harness even when"
                       " every other gate passes"))


    code, out = sh(f"PYTHONPATH=. python {generator}")
    ran = report.gate("samples", code == 0,
                      f"wrote {samples}" if code == 0 else generator + " failed\n" + tail(out),
                      fatal=True)
    if not ran:
        return report.failed
    body = samples.read_text().lower() if samples.is_file() else ""
    missing = [s for s in SECTIONS if s not in body]
    report.gate("sections", not missing,
                f"{samples.name} is missing the literal heading(s) {missing}"
                if missing else "all of Level 0, Level 2, Level 5, Answer present")

    digests = []
    for salt in ("0", "0", "1", "1", "2"):
        code, out = sh(f"PYTHONPATH=. python {generator}", {"PYTHONHASHSEED": salt})
        digests.append(hashlib.sha256(samples.read_bytes()).hexdigest()[:8] if code == 0 else "ERR")
    same_salt = digests[0] != digests[1] or digests[2] != digests[3]
    if "ERR" in digests and remaining() <= 1:
        report.gate("reproducible", None, "out of time: the generator is too slow to run"
                    " five times inside one command, so fix speed first")
    else:
        report.gate("reproducible", len(set(digests)) == 1, " ".join(digests) + (
        "" if len(set(digests)) == 1 else
        ("  -- two runs at the SAME salt disagree, so either the generator keeps state"
         " between calls or it iterates a dict/set keyed on objects, whose hash is their"
         " memory address and which PYTHONHASHSEED does not pin; key on a string or a"
         " tuple of ints instead" if same_salt else
         "  -- the same-salt pairs agreed, so this is most likely an unsorted set or"
         " dict of strings reaching the output: iterate sorted(...) over it. An"
         " object-keyed dict can also produce this pattern by chance, so if every"
         " set you render is already sorted, look for one keyed on objects.")))

    code, out = sh("python -m pytest -p no:cacheprovider --import-mode=importlib " + owned,
                   limit=90)
    report.gate("pytest", code == 0, "" if code == 0 else tail(out, 20))

    code, out = sh("python -c %s %s %d" % (
        shlex.quote(contract_audit_source()), shlex.quote(json.dumps(classes)),
        int(hashlib.sha256(trial.encode()).hexdigest()[:6], 16)), limit=min(120, remaining()))
    report.gate("contract", code == 0,
                "gold scores 1.0 and junk does not, over 64 examples" if code == 0
                else tail(out, 8))

    code, out = sh("python -m reasoning_core.task_search.prior_audit --path "
                   f"{owned} --n {args.n} --max-const 0.4 --budget-seconds 45")
    report.gate("gameability", code == 0, "" if code == 0 else tail(out, 10))
    print(f"\n{report.failed} gate(s) failing.", flush=True)
    return report.failed


if __name__ == "__main__":
    sys.exit(main())
