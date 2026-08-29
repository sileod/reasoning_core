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
"""
import argparse, ast, hashlib, json, os, pathlib, shlex, subprocess, sys

SECTIONS = ("level 0", "level 2", "level 5", "answer")


def contract_audit_source():
    """runner.py's contract audit, read from runner.py so the two cannot drift."""
    tree = ast.parse(pathlib.Path(__file__).with_name("runner.py").read_text())
    return ast.literal_eval(next(
        n.value for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_CONTRACT_AUDIT"))


def sh(command, env=None):
    """Run one shell command the way _run_validation does, from the worktree root."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {}))
    done = subprocess.run(["/bin/bash", "-c", command], capture_output=True,
                          text=True, env=environment)
    return done.returncode, (done.stdout + done.stderr)


def tail(text, n=12):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join("    | " + line for line in lines[-n:])


class Report:
    def __init__(self):
        self.failed, self.stop = 0, False

    def gate(self, name, ok, detail="", fatal=False):
        if ok is None:
            print(f"{name:<14} SKIP  {detail}")
            return False
        print(f"{name:<14} {'PASS' if ok else 'FAIL'}  {detail}".rstrip())
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

    code, out = sh(
        "python -c \"import importlib,random,sys,json;"
        "cs=json.loads(sys.argv[1]);"
        "[[t.config.set_level(L) or t.validate(n_samples=3) for L in (0,2,5)]"
        " for t in [getattr(importlib.import_module(m),c)() for m,c in cs]];"
        f"print('SMOKE_OK')\" '{json.dumps(classes)}'")
    if not report.gate("smoke", code == 0, "levels 0, 2, 5 validate" if code == 0
                       else "validate() failed\n" + tail(out), fatal=True):
        return report.failed

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

    code, out = sh("python -m pytest -p no:cacheprovider --import-mode=importlib " + owned)
    report.gate("pytest", code == 0, "" if code == 0 else tail(out, 20))

    code, out = sh("python -c %s %s %d" % (
        shlex.quote(contract_audit_source()), shlex.quote(json.dumps(classes)),
        int(hashlib.sha256(trial.encode()).hexdigest()[:6], 16)))
    report.gate("contract", code == 0,
                "gold scores 1.0 and junk does not, over 64 examples" if code == 0
                else tail(out, 8))

    code, out = sh("python -m reasoning_core.task_search.prior_audit --path "
                   f"{owned} --n {args.n} --max-const 0.4 --budget-seconds 45")
    report.gate("gameability", code == 0, "" if code == 0 else tail(out, 10))
    print(f"\n{report.failed} gate(s) failing.")
    return report.failed


if __name__ == "__main__":
    sys.exit(main())
