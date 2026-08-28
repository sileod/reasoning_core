# Evaluating task influence

Use the public influence pipeline to measure whether training on a new or changed
task improves held-out transfer against a shared baseline.

## Before running

1. Validate the task's generation and native scorer.
2. Materialize task rows once. Use the same rows for auxiliary training,
   begin/end reward, and saturation curves.
3. Pin remote models and datasets to exact 40-character revisions. Identify local
   inputs with `content_id()`.
4. Clone the model's initial `state_dict` once. `run_influence()` restores it before
   every arm.

Keep the baseline and treatment specs identical except for the treatment's `arm_id`
and auxiliary fields. To evaluate another task, normally change only:

- the task name or `StreamSpec.task` filter;
- the auxiliary data content ID;
- the treatment arm ID.

Set `shuffle_buffer=0` when comparing with the historical results. A positive value
is a deliberate protocol change. Put the battery and reward identifiers in
`ArmSpec.eval_ids`; changed scientific inputs then receive a new `spec_id`
automatically.

Put the versioned saturation identifier in `ArmSpec.callback_ids`. The canonical
callback records step zero and fixed training intervals, and resumes from its
arm-local curve sidecar after preemption.

## Run

Build one baseline `ArmPlan` and one treatment `ArmPlan`, then call
`run_influence()` with explicit metric names. Use `paper_battery()` for the frozen
paper battery or construct an `EvalBattery` from flexible `EvalLeg` entries. Pass a
small `FreeGenRewardSpec` evaluator through `evaluate_endpoints` to measure the same
task rows before and after training.

Start with a one- or two-step smoke. For the real run, preserve the model revision,
main data, token dose, seed, formatting, battery, and reward rows from the comparison
protocol.

## Interpret and report

`InfluenceResult.deltas` is `treatment - baseline`, so a negative NLL delta is
helpful. The public results use percentage NLL reduction:

```python
percent_nll_reduction = 100 * (baseline_nll - treatment_nll) / baseline_nll
```

Positive percentage reduction means the task helped. Report reward as
`initial → final`; it is diagnostic and is not part of the transfer score.

Record the task and source commit, immutable model/data/eval IDs, seed, token dose,
baseline and treatment metrics, percentage reductions, and begin/end reward. Never
combine results whose battery IDs differ into one ranking.

See the [training API](../reasoning_core/training/README.md) and
[reference results](RESULTS.md).
