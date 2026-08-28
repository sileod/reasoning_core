"""Constant-guess baseline: how much reward a task hands out for free.

A task that a single fixed answer wins is not measuring reasoning, however well it
validates. Point `--path` at a directory of task modules to gate a task_search trial,
or pass registered task names to audit what is already in DATASETS.
"""
import argparse, collections, importlib.util, inspect, pathlib, statistics, sys, time

LEVELS = (0, 3, 6)


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
        entries.append(task.generate())
        if time.time() > deadline:
            break
    answers = [str(e.answer) for e in entries]
    top, ntop = collections.Counter(answers).most_common(1)[0]
    return {
        "n": len(entries),
        "distinct": len(set(answers)) / len(answers),
        "const": statistics.mean(task.score_answer(top, e) for e in entries),
        "len": statistics.mean(map(len, answers)),
        "top": top,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="registered task names")
    ap.add_argument("--path", help="directory of task modules to import by file path")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-const", type=float, help="exit 1 if any level scores above this")
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

    worst = 0.0
    for name, task in tasks:
        for level in LEVELS:
            r = audit(task, level, args.n, time.time() + args.budget_seconds)
            # Too few samples to read a rate off; a slow generator should not fail here.
            if r["n"] >= 8:
                worst = max(worst, r["const"])
            print(f"{name:28s} L{level}  n={r['n']:3d}  distinct={r['distinct']:4.2f}"
                  f"  const_reward={r['const']:4.2f}  len={r['len']:5.1f}  ex={r['top'][:24]!r}")
    if args.max_const is not None and worst > args.max_const:
        print(f"FAIL: a constant guess scores {worst:.2f} > {args.max_const:.2f}; the task"
              " is winnable without reading the prompt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
