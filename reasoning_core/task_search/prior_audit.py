"""Two free-reward baselines: what a task hands out without being solved.

A fixed constant answer is one way to win without reasoning. Copying something off the
surface of the prompt is the other, and it is the one every mechanical gate misses: a
word problem that states its own total scores 1.00 here while passing determinism,
the contract, the constant-guess prior and pytest.

Point `--path` at a directory of task modules to gate a task_search trial, or pass
registered task names to audit what is already in DATASETS.
"""
import argparse, collections, importlib.util, inspect, pathlib, re, statistics, sys, time

LEVELS = (0, 3, 6)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def shortcuts(prompt):
    """Answers a reader could give from the prompt's surface, without solving anything.

    Deliberately crude and fixed: these are not clever attacks, they are what a model
    lands on when the prompt has already done the work. Reporting the best of them is
    the honest number, so the rules stay few and stay dumb.
    """
    numbers = NUMBER.findall(prompt or "")
    words = (prompt or "").split()
    return {
        "last_number": numbers[-1] if numbers else "",
        "first_number": numbers[0] if numbers else "",
        "largest_number": max(numbers, key=lambda n: abs(float(n))) if numbers else "",
        "last_word": words[-1].strip(".,:;?!") if words else "",
    }


def _load_from_path(root):
    """Task subclasses defined by the .py files under root, imported by file path."""
    from reasoning_core.template import Task
    found = []
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if path.name.startswith(("test_", "generate_samples_", "_")):
            continue
        spec = importlib.util.spec_from_file_location(f"_audit_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        found += [c for _, c in inspect.getmembers(module, inspect.isclass)
                  if issubclass(c, Task) and c is not Task and c.__module__ == spec.name]
    return found


def audit(task, level, n, deadline):
    task.config.set_level(level)
    entries = []
    for _ in range(n):
        # generate_example, not generate: only this path renders the prompt, and the
        # prompt is half of what is being audited.
        entries.append(task.generate_example())
        if time.time() > deadline:
            break
    answers = [str(e.answer) for e in entries]
    top, ntop = collections.Counter(answers).most_common(1)[0]
    rules = {name: statistics.mean(task.score_answer(shortcuts(e.prompt)[name], e)
                                   for e in entries)
             for name in shortcuts("")}
    worst_rule = max(rules, key=rules.get)
    return {
        "n": len(entries),
        "distinct": len(set(answers)) / len(answers),
        "const": statistics.mean(task.score_answer(top, e) for e in entries),
        "shortcut": rules[worst_rule],
        "rule": worst_rule,
        "len": statistics.mean(map(len, answers)),
        "top": top,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="registered task names")
    ap.add_argument("--path", help="directory of task modules to import by file path")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-const", type=float, help="exit 1 if any level scores above this")
    ap.add_argument("--max-shortcut", type=float,
                    help="exit 1 if reading an answer off the prompt scores above this")
    ap.add_argument("--budget-seconds", type=float, default=90.0,
                    help="stop sampling a level once this much time has gone into it")
    args = ap.parse_args(argv)

    if args.path:
        classes = _load_from_path(args.path)
        if not classes:
            print(f"no Task subclass under {args.path}", file=sys.stderr)
            return 1
        tasks = [(c.__name__, c()) for c in classes]
    else:
        import reasoning_core as rc
        tasks = [(name, rc.get_task(name)) for name in args.names]

    worst = {"const": 0.0, "shortcut": 0.0}
    worst_rule = ""
    for name, task in tasks:
        for level in LEVELS:
            r = audit(task, level, args.n, time.time() + args.budget_seconds)
            # Too few samples to read a rate off; a slow generator should not fail here.
            if r["n"] >= 8:
                worst["const"] = max(worst["const"], r["const"])
                if r["shortcut"] > worst["shortcut"]:
                    worst["shortcut"], worst_rule = r["shortcut"], r["rule"]
            print(f"{name:28s} L{level}  n={r['n']:3d}  distinct={r['distinct']:4.2f}"
                  f"  const_reward={r['const']:4.2f}  shortcut={r['shortcut']:4.2f}"
                  f" ({r['rule']})  len={r['len']:5.1f}  ex={r['top'][:24]!r}")
    if args.max_const is not None and worst["const"] > args.max_const:
        print(f"FAIL: a constant guess scores {worst['const']:.2f} > {args.max_const:.2f};"
              " the task is winnable without reading the prompt", file=sys.stderr)
        return 1
    if args.max_shortcut is not None and worst["shortcut"] > args.max_shortcut:
        print(f"FAIL: answering with the {worst_rule} in the prompt scores"
              f" {worst['shortcut']:.2f} > {args.max_shortcut:.2f}; the prompt states its"
              " own answer, so the task is extraction and not reasoning", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
