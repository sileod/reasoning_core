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

Start with the [complete offline smoke example](workflows.md#run-a-paired-influence-smoke).
It supplies the model, tokenizer, dataset factories, evaluation callback, and immutable IDs.
Use the protocol below when replacing its toy inputs with research data.

Build one baseline `ArmPlan` and one treatment `ArmPlan`, then call
`run_influence()` with explicit metric names. Use `default_battery()` — the 39-leg
battery current results are measured on — or construct an `EvalBattery` from flexible
`EvalLeg` entries. `paper_battery()` is the legacy 21-leg battery, kept only to
reproduce the first paper's numbers; the two cannot be pooled.

A battery's identifier covers `max_length`, so build it at the same `max_length` you
train at. Published results use 1024. Pass a
small `FreeGenRewardSpec` evaluator through `evaluate_endpoints` to measure the same
task rows before and after training.

## The protocol to match

To produce a number comparable to the published results, pin all of it:

| | |
|---|---|
| battery | `default_battery()`, built at `max_length=1024` |
| main stream | identified by `content_id()`, never by path alone |
| model | an exact 40-character revision |
| dose | auxiliary share fixed by tokens, not example count |

Batch size, gradient accumulation, optimizer and learning-rate schedule are part of the
measurement too: two runs that differ on any of them are not poolable, and a report that
averages them ranks nothing.

Cluster orchestration -- allocation, preemption, scratch, resume -- is site-specific and is
not part of this package. Run locally with the linked smoke recipe; wrap it in whatever your site
uses for batch jobs, running the arm in the foreground so the job's exit status is the run's.

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

See the [training API](evaluation.md) and
[reference results](results/influence.md).
