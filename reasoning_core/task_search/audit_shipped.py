"""Run the coordinator's contract audit against every task the library ships.

The gates that judge a candidate have no independent calibration: a gate can be strict,
or it can be wrong, and a wave of failed trials alone cannot tell those apart. The tasks
already in the library are the calibration. A task that ships and cannot pass a gate is
evidence about the gate, not about the task, and the gate is what changes.

It reads the other way too, as a gameability sweep over what already ships. The first run
found MetamathCoreSelect scoring "reajrjrje9595!" as correct against gold "A", because a
bare [A-D] search matches the letter inside any word.

    python -m reasoning_core.task_search.audit_shipped
"""
import argparse
import random

import reasoning_core
import reasoning_core.template

JUNK = ("", " ", "reajrjrje9595!")

_base_validate = reasoning_core.template.Task.validate
_reached = []


def _spy_validate(self, *args, **kwargs):
    _reached.append(type(self).__name__)
    return _base_validate(self, *args, **kwargs)


def audit(task_class, *, samples, seed):
    """The candidate contract, applied to one shipped class. Empty means it passed."""
    task = task_class()
    random.seed(seed)
    del _reached[:]
    task.validate(n_samples=4)
    if not _reached:
        # The spy is installed after import, so a module that reassigns Task.validate
        # replaces the checker rather than extending it, and never reaches this.
        return "replaced Task.validate instead of extending it"
    for index in range(samples):
        entry = task.generate_example()
        if task.score_answer(entry.answer, entry) != 1:
            return f"gold answer rejected (sample {index})"
        for junk in JUNK:
            if junk != str(entry.answer) and task.score_answer(junk, entry) >= 1:
                return f"junk {junk!r} scored correct against gold {entry.answer!r}"
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--task", action="append", default=[],
                        help="audit only these registered task names")
    args = parser.parse_args(argv)

    names = args.task or reasoning_core.list_tasks()
    reasoning_core.template.Task.validate = _spy_validate
    failures = []
    print(f"auditing {len(names)} shipped task classes", flush=True)
    for name in names:
        try:
            problem = audit(type(reasoning_core.get_task(name)),
                            samples=args.samples, seed=args.seed)
        except Exception as error:            # a task that cannot run is a failure too
            problem = f"{type(error).__name__}: {str(error)[:160]}"
        if problem:
            failures.append((name, problem))
    print(f"\n{len(names) - len(failures)}/{len(names)} shipped tasks pass the contract audit")
    for name, problem in failures:
        print(f"  FAIL {name}: {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
