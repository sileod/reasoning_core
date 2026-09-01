"""Command-line interface for task search."""

import argparse
import json
import os
from pathlib import Path
import sys

from .implementor_prompt import DEFAULT_PACE, PACE, render_implementor_prompt
from .plan import _frozen_module_drift, _plan_problems, load_plan
from .runner import _repo_root, run_plan
from .sandbox import _write_json


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate and summarize a plan")
    check.add_argument("plan")
    render = subparsers.add_parser("render", help="render one worker prompt")
    render.add_argument("plan")
    render.add_argument("trial_id")
    render.add_argument("--pace", choices=sorted(PACE), default=DEFAULT_PACE)
    propose = subparsers.add_parser(
        "propose", help="generate an SFT-first proposal wave with novelty review"
    )
    propose.add_argument("name")
    propose.add_argument("--count", type=int, default=12)
    propose.add_argument("--output")
    propose.add_argument("--model", default="moonshotai/kimi-k3")
    propose.add_argument(
        "--endpoint", default="https://integrate.api.nvidia.com/v1/chat/completions"
    )
    propose.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    propose.add_argument("--seed", type=int, default=0)
    propose.add_argument("--temperature", type=float, default=1.0)
    propose.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="max"
    )
    propose.add_argument("--rounds", type=int, default=3)
    propose.add_argument("--max-catalog-chars", type=int, default=240000)
    proposal_check = subparsers.add_parser(
        "check-proposals", help="validate an archived SFT proposal wave"
    )
    proposal_check.add_argument("proposal_wave")
    catalog = subparsers.add_parser(
        "proposal-catalog", help="summarize the durable novelty catalog"
    )
    catalog.add_argument(
        "--output", help="optionally write the complete catalog as JSON"
    )
    run = subparsers.add_parser("run", help="launch folder-scoped coding workers")
    run.add_argument("plan")
    run.add_argument("--model", required=True)
    run.add_argument(
        "--harness", choices=("opencode", "mini", "agy"), default="opencode"
    )
    run.add_argument("--provider", help="optional Harness Link provider")
    run.add_argument(
        "--credential-env",
        action="append",
        default=[],
        help="environment variable to remove from candidate validation processes",
    )
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--trial", action="append", default=[])
    run.add_argument("--queue", action="append", default=[])
    run.add_argument("--agent", default="task-search-worker")
    run.add_argument("--variant")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--forward-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forward each derived trial seed to OpenCode (default: enabled)",
    )
    run.add_argument("--temperature", type=float)
    run.add_argument("--top-p", type=float)
    run.add_argument("--max-steps", type=int, default=56)
    run.add_argument(
        "--pace",
        choices=sorted(PACE),
        default=DEFAULT_PACE,
        help="how hard the worker is told to hurry; recorded in generation metadata so waves stay comparable",
    )
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--transient-retries", type=int, default=2)
    run.add_argument("--retry-backoff-seconds", type=int, default=30)
    run.add_argument("--validation-timeout-seconds", type=int, default=300)
    run.add_argument("--hlink-bin", default="hlink")
    run.add_argument("--bwrap-bin", default="bwrap")
    run.add_argument(
        "--resource-limits",
        choices=("auto", "required", "none"),
        default="auto",
        help="apply a user systemd scope to every worker and validation process",
    )
    run.add_argument("--systemd-run-bin", default="systemd-run")
    run.add_argument("--memory-max", default="8G")
    run.add_argument("--tasks-max", type=int, default=512)
    run.add_argument("--cpu-quota", default="400%")
    run.add_argument("--runs-root")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "propose":
        from .wave_proposer import propose_wave, write_proposal_wave

        repo_root = _repo_root(Path.cwd())
        output = (
            Path(args.output)
            if args.output
            else repo_root
            / "reasoning_core"
            / "task_search"
            / "proposals"
            / "archive"
            / f"{args.name}.yaml"
        )
        if output.exists():
            raise SystemExit(
                f"refusing to spend model calls: archive already exists: {output}"
            )
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"{args.api_key_env} is required for the NVIDIA NIM proposer"
            )
        wave = propose_wave(
            repo_root,
            name=args.name,
            count=args.count,
            model=args.model,
            endpoint=args.endpoint,
            api_key=api_key,
            seed=args.seed,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            rounds=args.rounds,
            max_catalog_chars=args.max_catalog_chars,
        )
        write_proposal_wave(output, wave)
        print(
            f"{output}: {len(wave['proposals'])} accepted, "
            f"{len(wave['rejected'])} rejected"
        )
        if not wave["objective"]["complete"]:
            print(
                f"INCOMPLETE: requested {wave['objective']['requested']}; "
                "increase --rounds after reviewing the archive",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return
    if args.command == "check-proposals":
        from .wave_proposer import check_proposal_file

        problems = check_proposal_file(args.proposal_wave)
        if problems:
            for problem in problems:
                print(f"PROBLEM: {problem}")
            raise SystemExit(1)
        print(f"{args.proposal_wave}: OK")
        return
    if args.command == "proposal-catalog":
        from .wave_proposer import build_catalog, catalog_record

        repo_root = _repo_root(Path.cwd())
        entries = build_catalog(repo_root)
        record = catalog_record(entries)
        print(json.dumps(record, indent=2, sort_keys=True))
        if args.output:
            _write_json(Path(args.output), [entry.as_dict() for entry in entries])
        return

    plan = load_plan(args.plan)
    if args.command == "check":
        print(f"{plan.name}: {len(plan.trials)} trials from {plan.base_ref}")
        repo_root = _repo_root(plan.path.parent)
        for problem in _plan_problems(plan, repo_root):
            print(f"PROBLEM: {problem}")
        drift = _frozen_module_drift(repo_root, plan.base_ref)
        if drift:
            print(f"WARNING: {drift}")
        for name, members in plan.queues.items():
            print(f"queue\t{name}\t{','.join(members)}")
        for trial in plan.trials:
            print(f"{trial.trial_id}\t{trial.hypothesis or '-'}\t{trial.owned_path}")
    elif args.command == "render":
        trial = next(
            (item for item in plan.trials if item.trial_id == args.trial_id), None
        )
        if trial is None:
            raise SystemExit(f"unknown trial: {args.trial_id}")
        # A template preview: execution builds a model-, seed- and budget-dependent
        # TASK_META and passes it in. The prompt a worker actually got is its
        # prompt.md, in the trial directory.
        print(
            render_implementor_prompt(
                plan, trial, _repo_root(plan.path.parent), pace=args.pace
            ),
            end="",
        )
    else:
        results = run_plan(
            args.plan,
            model=args.model,
            jobs=args.jobs,
            trial_ids=args.trial,
            queue_names=args.queue,
            agent=args.agent,
            variant=args.variant,
            seed=args.seed,
            forward_seed=args.forward_seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_steps=args.max_steps,
            timeout_seconds=args.timeout_seconds,
            transient_retries=args.transient_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            harness=args.harness,
            hlink_bin=args.hlink_bin,
            bwrap_bin=args.bwrap_bin,
            runs_root=args.runs_root,
            provider=args.provider,
            resource_limit_mode=args.resource_limits,
            systemd_run_bin=args.systemd_run_bin,
            memory_max=args.memory_max,
            tasks_max=args.tasks_max,
            cpu_quota=args.cpu_quota,
            validation_timeout_seconds=args.validation_timeout_seconds,
            credential_env_names=args.credential_env,
            pace=args.pace,
        )
        for result in results:
            print(
                f"{result['trial_id']}\t{result['status']}\t"
                f"{result.get('worktree', '-')}"
            )
        if any(result["status"] != "success" for result in results):
            raise SystemExit(1)
