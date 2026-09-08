# Task Search

`reasoning_core.task_search` turns reviewed task ideas into isolated implementation
trials. Proposal generation remains a separate subsystem in `wave_proposer.py`.

## Package boundaries

- `plan.py` owns the frozen `Trial` and `SearchPlan` models, YAML loading, plan checks,
  and trial selection.
- `implementor_prompt.py` owns `PACE` and `render_implementor_prompt()`.
- `sandbox.py` owns Bubblewrap, resource limits, sanitized environments, and bounded
  validation subprocesses.
- `validation.py` owns every coordinator and worker-facing gate and their failure
  precedence. `selfcheck.py` is only its CLI compatibility wrapper.
- `implementation_runner.py` creates worktrees, launches Harness Link, validates candidates, retries
  explicit infrastructure failures, and writes run records.
- `cli.py` contains argument parsing and command dispatch.
- `plans/` holds the `wave*.yaml` plans and the idea documents they came from;
  `proposals/` holds proposal waves and the archive a plan is generated from.

## Run a wave

Install a Harness Link version that provides the `hlink` frontend, then validate,
render, or run a plan:

```bash
python -m reasoning_core.task_search check reasoning_core/task_search/plans/wave8.yaml
python -m reasoning_core.task_search render reasoning_core/task_search/plans/wave8.yaml P001v1
python -m reasoning_core.task_search run reasoning_core/task_search/plans/wave8.yaml \
  --harness opencode --model deepseek-v4-flash --provider albert --trial P001v1
```

`--provider` is optional. Harness Link owns harness discovery, provider adaptation,
model selection, cwd, prompt delivery, unattended mode, and native argument forwarding.
Task search still owns experiment-specific permissions/configuration, step limits,
trajectory paths, AGY writable overlays, and the outer sandbox/resource limits.

`run` requires a plan and `--model`. Defaults are OpenCode, one job, 56 steps,
and a 30-minute worker timeout. With no `--trial` or `--queue`, it runs every
trial in the plan; use `--trial ID` for a single-task smoke run.

OpenCode filesystem snapshots are disabled by default to avoid a second Git index
of each isolated worktree. Use `--snapshots` to enable session undo, or
`--no-snapshots` to disable it explicitly. This setting affects only OpenCode and
is recorded in `summary.json` and `run.json`; it does not change the worker prompt,
sandbox permissions, validation, or retained worktrees.

Each trial runs from the same detached base commit and may write only its owned task
directory and private runtime. The coordinator independently checks scope, provenance,
discovery, contract behavior, samples, reproducibility, validation commands,
gameability, semantics, and candidate stability in a fixed order.

Runs default to `.reasoning_core-task-search/<wave>/<timestamp>/` beside the checkout.
They retain prompts, harness output, validation logs, candidate hashes, `run.json`, and
an incrementally updated `summary.json`. See `BABYSITTING.md` for safe monitoring.
The runner prints the artifact directory at startup. Custom `--runs-root` paths
must be outside `/tmp` and `/run`, which are hidden by the sandbox.

## Before you launch

```bash
python -m reasoning_core.task_search doctor --live
```

It checks the two credential paths, which are separate and both fail open. Worker
credentials reach the coding agent through a copy of the environment, so the provider's
own key (`ALBERT_API_KEY` for albert) must be set in the shell that launches the run; a
wave without it spends its whole queue on `harness_failed`. The semantic reviewer reads
`TASK_SEARCH_REVIEW_ENDPOINT`, `TASK_SEARCH_REVIEW_MODEL` and
`TASK_SEARCH_REVIEW_KEY_ENV` instead; without them it returns a null verdict for every
trial and `land` skips them all as `unreviewed`. Both live in
`~/.config/reasoning_core/env`, which is outside the checkout and must be sourced --
background scripts do not inherit it. `--live` spends one tiny completion to prove the
key is not merely present but accepted, which is how a spent daily quota shows up before
a run rather than during one.

`run` defaults `--model` to `deepseek-v4-flash`, the implementor every landed wave
was built with. It has no default provider: which host serves that model is a fact
about a machine, so set `TASK_SEARCH_PROVIDER` in the env file and `run` and `doctor`
both pick it up.

## Propose tasks

```bash
python -m reasoning_core.task_search proposal-catalog
python -m reasoning_core.task_search propose sft-wave-1 --count 12
python -m reasoning_core.task_search check-proposals \
  reasoning_core/task_search/proposals/archive/sft-wave-1.yaml
```

See `proposals/FORMAT.md` for the proposal schema and novelty rules.

## Compare implementation choices

Generate two distinct approaches per proposal, then run one worker per approach:

```bash
python -m reasoning_core.task_search plan proposals.yaml --name choice_pilot \
  --variants 2 --design-choices 2
python -m reasoning_core.task_search run reasoning_core/task_search/plans/choice_pilot.yaml \
  --model deepseek-v4-flash --provider albert
```

The design proposer defaults to DeepSeek V4 Flash on Albert and uses `ALBERT_API_KEY`.
`--design-choices` must equal `--variants`; without it, variants differ only by seed.
Every proposal in the input is included, so use a proposal subset for a small pilot.

The exact assigned approach is stored as `trials[].design_choice` in the plan, under
`Assigned design choice` in `prompt.md`, and as `design_choice` in each completed
trial's `run.json` and its `summary.json` result. This records the assignment, not
proof that the implementation followed it: inspect the candidate and samples before
comparing approaches. The generated module's `TASK_META` does not include this field;
retain the plan and run records alongside any selected candidate.
