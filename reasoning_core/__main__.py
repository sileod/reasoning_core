"""Discover tasks, generate examples, validate generators, and score predictions."""

import argparse
import json
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="search names, summaries, and source paths without imports")
    catalog.add_argument("query", nargs="?", default="")
    catalog.add_argument("--all", action="store_true", help="include generated, mutated, and discovered DevTask classes")
    catalog.add_argument("--include-generated", action="store_true")
    catalog.add_argument("--json", action="store_true")
    sample = commands.add_parser("sample", help="write scored reference examples as JSONL")
    sample.add_argument("task")
    sample.add_argument("--count", type=positive_int, default=3)
    sample.add_argument("--level", type=float, default=0)
    sample.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="run the task authoring contract")
    validate.add_argument("task")
    validate.add_argument("--samples", type=positive_int, default=10)
    validate.add_argument("--level", type=float, default=0)
    score = commands.add_parser("score", help="score JSONL rows containing a prediction field")
    score.add_argument("input", type=Path)
    args = parser.parse_args(argv)

    from reasoning_core import get_task, score_answer, task_catalog

    if args.command == "catalog":
        rows = task_catalog(args.query, include_generated=args.all or args.include_generated,
                            include_mutated=args.all, include_dev=args.all)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['name']} [{row['status']}/{row['origin']}] — {row['summary']}"
                      f"\n  {row['source']}:{row['line']}")
    elif args.command == "sample":
        task = get_task(args.task)
        # Generate and serialize before publishing; refuse to overwrite an existing dataset.
        rows = []
        for _ in range(args.count):
            example = task.generate_example(level=args.level)
            if score_answer(example.answer, example) != 1:
                raise RuntimeError(f"{args.task}: reference answer did not score 1")
            rows.append(json.dumps(example.to_dict()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write("\n".join(rows) + "\n")
        print(json.dumps({"output": str(args.output), "examples": len(rows)}))
    elif args.command == "validate":
        task = get_task(args.task)
        task.config.set_level(args.level)
        task.validate(n_samples=args.samples)
        print(json.dumps({"task": task.task_name, "level": args.level, "valid": True}))
    else:
        from easydict import EasyDict
        scores = []
        for line_number, line in enumerate(args.input.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = EasyDict(json.loads(line))
            if "prediction" not in row:
                parser.error(f"{args.input}:{line_number}: missing prediction field")
            scores.append(float(score_answer(row.prediction, row)))
        if not scores:
            parser.error("input contains no examples")
        print(json.dumps({"examples": len(scores), "mean_score": sum(scores) / len(scores),
                          "scores": scores}))


if __name__ == "__main__":
    main()
