# Training and influence

The public training API has one arm runner:

- `arm.py`: typed, resumable, content-addressed `run_arm()`
- `data.py`: explicit formatting, filtering, mixing, and token-dose helpers
- `evals.py`: versioned QA, LM, multiple-choice, and generation evaluators
- `influence.py`: paired baseline/treatment orchestration using `run_arm()`
- `gradient_influence.py`: cached contrastive gradients and cheap task alignment

Install the optional dependencies with:

```bash
pip install 'reasoning-core[training]'
```

## Eval data

The battery legs ship in `eval_data/battery_legs.zip` and unpack themselves into `data_cache/`
(override with `EVAL_DATA_DIR`) the first time a battery loads. They are shipped rather than rebuilt
because a leg's identity is the sha256 of its bytes: a regenerated leg is a *different* leg, and
results across differing battery IDs must never be pooled. The archive lives outside the package, so
it is in the git repo but not in the PyPI wheel or sdist.

An influence experiment is a baseline `ArmPlan` plus one or more treatment
plans. Dataset factories are called independently, and the same supplied model
state is cloned once and restored before every arm:

```python
from reasoning_core.training.arm import ArmSpec
from reasoning_core.training.influence import ArmPlan, run_influence

baseline = ArmPlan(ArmSpec(
    "study", "baseline", initialization_id="sha256:...", main_data_id="sha256:...",
    eval_ids=("heldout/answer_nll@v1:...",),
), main_data)
treatment = ArmPlan(
    ArmSpec(
        "study", "task-x", initialization_id="sha256:...", main_data_id="sha256:...",
        aux_source="task-x", aux_data_id="sha256:...", aux_fraction=0.2,
        eval_ids=("heldout/answer_nll@v1:...",),
    ),
    mixed_data,
)
result = run_influence(
    model, tokenizer, initial_state, baseline, (treatment,),
    metric_names=("nll",), evaluate=evaluate,
)
print(result.deltas)
```

`ArmSpec.spec_id` covers the engine version and complete serialized spec. Status files additionally
record engine, package, dependency, initialization, data, and evaluation IDs.
Arm construction rejects missing initialization and data IDs. Use
`data.content_id()` for local files/directories and provide pinned revision IDs
for remote inputs. `data.source_id()` accepts only exact 40-character Hub commits,
which callers must also pass as the model/dataset loader revision. External
callbacks likewise require matching version IDs in `ArmSpec.callback_ids`.

Benchmark batteries are ordered data, not runner logic. Build any battery from
`EvalLeg` objects or `load_battery_manifest()`; `paper_battery()` is only the
shipped paper default. Its BBH dev and BBH dev-cloze entries share frozen examples
but remain separate, reorderable legs; the held-out test split likewise has raw,
cloze, and options-omitted legs. MCQ legs emit paired `<name>_nll`,
`<name>_mc_cloze_acc`, and margin metrics from one scoring pass. Record
`battery.identifier` in `ArmSpec.eval_ids`.

`FreeGenRewardSpec` in `intrinsic_rewards.py` configures native task reward without
environment variables. Pass a small reward evaluator as `evaluate_endpoints` to
`run_influence()` to record the shared initial reward and each arm's final reward,
or attach it to one treatment with `ArmPlan.evaluate_endpoint`.

`SaturationCurveSpec` in `saturation.py` configures periodic teacher-forced
answer-token accuracy. Attach a `SaturationCurveCallback` to a treatment plan and
record its `saturation_id` in `ArmSpec.callback_ids`. Curves are batched, written
atomically under the arm directory, recovered after checkpoint resume, and included
in the completed arm metrics.

`StreamSpec` reads local JSON/JSONL/Parquet (including Parquet directories) or a
streaming Hub dataset. Remote models and datasets require exact commit revisions;
local inputs are identified by their content hashes.

Ordering is never implicit: every `mix_streams()` call must choose a
`shuffle_buffer`. Use `0` to reproduce the legacy influence protocol. Enabling a
positive deterministic shuffle is a protocol change and produces a different
arm identity when recorded in `ArmSpec`.

The former `dev_*` modules remain compatibility imports. New code should import
the canonical modules above.

For a new or changed task, follow the repository's
[task influence guide](../../task_influence/README.md).
It keeps the baseline fixed and limits task-specific changes to the auxiliary
stream/filter, its content ID, and the treatment arm ID. Historical measurements
are published in [`RESULTS.md`](../../task_influence/RESULTS.md).

## Gradient influence proxy

The gradient proxy is separate from paired training: it has no trainer, optimizer,
checkpoint, callback, `run_arm()`, or changes to model weights. It builds one
content-addressed aggregate from ordered MC benchmark legs, then scores formatted
`prompt`/`completion` batches:

```python
from reasoning_core.training.gradient_influence import (
    GradientCacheSpec, build_eval_gradient_cache, gradient_objective_id,
    score_task_gradient,
)

objective_id = gradient_objective_id(legs, max_length=512)
spec = GradientCacheSpec("sha256:warmed-state", objective_id, 512)
cache = build_eval_gradient_cache(model, tokenizer, legs, spec)
result = score_task_gradient(model, tokenizer, rows, cache, max_length=512)
print(result.cosine, result.dot, result.task_norm)
```

Each benchmark leg contributes its unit-normalized gradient. The final aggregate
is normalized and stored as safetensors plus a provenance manifest under
`~/.cache/reasoning_core/gradient_influence/`. `score_task()` accepts a seeded
batch factory and returns independent mean/std/stderr plus aggregate cosine.

Run `scripts/validate_gradient_influence.py --help` to calibrate 1/2/4/8/16/32
batch estimates against the published 51-task, 300-step measurements. Its pinned
5M default is a cheap pipeline smoke; pass the actual warmed FW+Dolci checkpoint
and initialization ID for scientific calibration.
