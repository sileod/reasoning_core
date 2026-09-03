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

Each trial runs from the same detached base commit and may write only its owned task
directory and private runtime. The coordinator independently checks scope, provenance,
discovery, contract behavior, samples, reproducibility, validation commands,
gameability, semantics, and candidate stability in a fixed order.

Runs default to `.reasoning_core-task-search/<wave>/<timestamp>/` beside the checkout.
They retain prompts, harness output, validation logs, candidate hashes, `run.json`, and
an incrementally updated `summary.json`. See `BABYSITTING.md` for safe monitoring.

## Propose tasks

```bash
python -m reasoning_core.task_search proposal-catalog
python -m reasoning_core.task_search propose sft-wave-1 --count 12
python -m reasoning_core.task_search check-proposals \
  reasoning_core/task_search/proposals/archive/sft-wave-1.yaml
```

See `proposals/FORMAT.md` for the proposal schema and novelty rules.
