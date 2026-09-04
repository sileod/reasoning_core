"""Two free-reward baselines: what a task hands out without being solved.

The constant-guess rate is measured on the balanced label distribution the dataset
is actually built with, and reported as the excess over the 1/k floor every k-label
task must give away. Both corrections matter: the raw rate on the raw draw rejected
13 of 70 shipped tasks, 12 of them for a skew that the balancing cap erases.

A fixed constant answer is one way to win without reasoning. Copying something off the
surface of the prompt is the other, and it is the one every mechanical gate misses: a
word problem that states its own total scores 1.00 here while passing determinism,
the contract, the constant-guess prior and pytest.

Point `--path` at a directory of task modules to gate a task_search trial, or pass
registered task names to audit what is already in DATASETS.
"""
import argparse, collections, importlib.util, inspect, math, pathlib, re, statistics, sys, time

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


def _balanced(entries, task, n):
    """The subset generate_balanced_batch would have kept -- the distribution that ships.

    Datasets are built by generate_balanced_batch, which caps any one balancing_key
    (the answer string, by default) at ceil(batch_size * balancing_key_ratio). Auditing
    the raw generate_example draw measures a label skew no model ever sees: five shipped
    tasks read 0.53-0.70 raw and land at exactly 0.50 once the cap is applied. Filtering
    the draw we already have costs no extra generation and cannot blow the deadline.
    """
    cap = math.ceil(n * getattr(task, "balancing_key_ratio", 0.5))
    counts, kept = collections.Counter(), []
    for entry in entries:
        key = getattr(entry, "balancing_key", None)
        if key is not None:
            if counts[key] >= cap:
                continue
            counts[key] += 1
        kept.append(entry)
    return kept


def audit(task, level, n, deadline):
    task.config.set_level(level)
    entries = []
    error = ""
    for _ in range(n):
        try:
            # generate_example, not generate: only this path renders the prompt, and
            # the prompt is half of what is being audited. It also carries template's
            # per-example timeout, which is how a level nobody can generate at shows
            # up here -- the speed gate only ever times the default config.
            entries.append(task.generate_example())
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:120]
            break
        if time.time() > deadline:
            break
    if not entries:
        return {"n": 0, "error": error or "no examples generated", "distinct": 0.0,
                "const": 0.0, "excess": 0.0, "shortcut": 0.0, "rule": "-", "len": 0.0,
                "top": ""}
    balanced = _balanced(entries, task, n)
    answers = [str(e.answer) for e in balanced]
    top, ntop = collections.Counter(answers).most_common(1)[0]
    rules = {name: statistics.mean(task.score_answer(shortcuts(e.prompt)[name], e)
                                   for e in entries)
             for name in shortcuts("")}
    worst_rule = max(rules, key=rules.get)
    const = statistics.mean(task.score_answer(top, e) for e in balanced)
    # A k-label task hands a constant guesser 1/k however good it is, so the raw rate
    # reads label-set size, not gameability: the pool's best task (regex_reasoning,
    # 8 labels) is further above its floor than the binary tasks the raw rate rejects.
    # Only the excess over that floor is avoidable, so only the excess is gateable.
    return {
        "n": len(balanced),
        "error": error,
        "distinct": len(set(answers)) / len(answers),
        "const": const,
        # k == 1 is a level that only ever emits one answer. Its 1/k floor is 1.00, so
        # subtracting it would score a wholly constant level at zero and pass it; the
        # unsolvable case has to report the raw rate instead.
        "excess": (const - 1.0 / len(set(answers))) if len(set(answers)) > 1 else const,
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
    ap.add_argument("--max-const", type=float,
                    help="exit 1 if a constant guess beats the 1/k floor by more than this,"
                         " measured on the balanced (as-shipped) label distribution")
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

    worst = {"excess": 0.0, "shortcut": 0.0}
    worst_rule = ""
    broken = []
    for name, task in tasks:
        for level in LEVELS:
            r = audit(task, level, args.n, time.time() + args.budget_seconds)
            if r["error"]:
                # A level the generator cannot survive is a broken level, whatever the
                # rates say; the sweep is the only place it is ever exercised.
                broken.append(f"level {level}: {r['error']}")
            # Too few samples to read a rate off; a slow generator should not fail here.
            counted = r["n"] >= 8
            if counted:
                worst["excess"] = max(worst["excess"], r["excess"])
                if r["shortcut"] > worst["shortcut"]:
                    worst["shortcut"], worst_rule = r["shortcut"], r["rule"]
            print(f"{name:28s} L{level}  n={r['n']:3d}  distinct={r['distinct']:4.2f}"
                  f"  const_reward={r['const']:4.2f}  excess={r['excess']:+5.2f}"
                  f"  shortcut={r['shortcut']:4.2f}"
                  f" ({r['rule']})  len={r['len']:5.1f}  ex={r['top'][:24]!r}"
                  # n<8 is too few to read a rate off, so the level is printed but not
                  # gated. Saying so stops a 0.50 off two unique answers reading as a
                  # verdict -- it is arithmetic, not evidence.
                  + ("" if counted else "  [n<8, not gated]")
                  + (f"  BROKEN {r['error']}" if r["error"] else ""))
    if broken and args.max_const is not None:
        print("FAIL: generation does not survive every level:\n  " + "\n  ".join(broken),
              file=sys.stderr)
        return 1
    if args.max_const is not None and worst["excess"] > args.max_const:
        print(f"FAIL: a constant guess beats the 1/k floor by {worst['excess']:.2f} >"
              f" {args.max_const:.2f} on the balanced distribution; the label prior is"
              " skewed enough to win without reading the prompt", file=sys.stderr)
        return 1
    if args.max_shortcut is not None and worst["shortcut"] > args.max_shortcut:
        print(f"FAIL: answering with the {worst_rule} in the prompt scores"
              f" {worst['shortcut']:.2f} > {args.max_shortcut:.2f}; the prompt states its"
              " own answer, so the task is extraction and not reasoning", file=sys.stderr)
        return 1
    if args.max_const is not None:
        # Passing is not the same as passing comfortably, and only this line tells the
        # author which of the two they did.
        print(f"OK: worst constant-guess excess {worst['excess']:+.2f}, worst prompt-surface guess"
              f" {worst['shortcut']:.2f} ({worst_rule or '-'}), ceiling {args.max_const:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
