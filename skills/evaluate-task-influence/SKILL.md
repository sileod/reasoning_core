---
name: evaluate-task-influence
description: Evaluate or re-evaluate a Reasoning Core task with the public paired influence pipeline. Use when adding or changing a task, measuring held-out transfer against a shared baseline, checking task-native reward before and after training, or updating an influence result report.
---

# Evaluate Task Influence

Measure one task by changing only its auxiliary data and comparing it with a
content-addressed baseline from identical model weights.

## Prepare

1. Read `TASK_AUTHORING_GUIDE.md` before changing a task.
2. Read `reasoning_core/training/README.md` for the public API.
3. Read `INFLUENCE_RESULTS.md` only when comparing with the historical protocol.
4. Check the worktree and preserve unrelated changes.
5. Validate the task's generation and native scorer before spending GPU time.

Keep migration adapters, collection rosters, and legacy runners private. Use only
`reasoning_core.training` in public callers.

## Freeze the experiment

- Pin remote models and datasets to exact 40-character revisions.
- Identify local files or directories with `content_id()`.
- Materialize task rows once and use the same rows for auxiliary training and
  begin/end reward.
- Clone the model's initial `state_dict` once. `run_influence()` restores it before
  every arm.
- Keep baseline and treatment specs identical except for `arm_id` and auxiliary
  fields.
- Set `shuffle_buffer=0` when comparing with the historical results. A positive
  value is a deliberate protocol change.
- Put the battery identifier and reward identifier in `ArmSpec.eval_ids`.

For a newly added task, normally change only:

1. the task name or `StreamSpec.task` filter;
2. the auxiliary data content ID;
3. the treatment arm ID.

Do not delete old run directories to force a rerun. Changed scientific inputs
produce a new `spec_id` automatically.

## Run

Build one baseline `ArmPlan` and one treatment `ArmPlan`, then call
`run_influence()` with explicit metric names. Use `paper_battery()` for the frozen
paper battery or construct an `EvalBattery` from flexible `EvalLeg` entries. Pass a
small `FreeGenRewardSpec` evaluator through `evaluate_endpoints` to measure the same
task rows before and after training.

Start with a tiny one- or two-step smoke. For the real run, preserve model revision,
main data, token dose, seed, formatting, battery, and reward rows from the comparison
protocol.

## Interpret and report

`InfluenceResult.deltas` is `treatment - baseline`. Negative NLL delta is helpful.
For the public percentage convention, report:

```python
percent_nll_reduction = 100 * (baseline_nll - treatment_nll) / baseline_nll
```

Positive percentage reduction means the task helped. Reward is diagnostic and
should be reported as `initial → final`, not folded into the transfer score.

Update a Markdown report with the task and source commit, immutable model/data/eval
IDs, seed, token dose, baseline and treatment metrics, percentage reductions, and
begin/end reward. State whether the run matches the historical protocol or changes
it. Never combine results whose battery IDs differ into one ranking.

## Verify

- Run focused task and training tests.
- Confirm every arm records provenance and a distinct content-addressed directory.
- Confirm the shared baseline and treatment begin from identical weights.
- Treat the legacy runner only as a private oracle; do not add public dependencies
  on it.
