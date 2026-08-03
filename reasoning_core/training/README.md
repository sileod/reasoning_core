# Training and influence

The public training API has one arm runner:

- `arm.py`: typed, resumable, content-addressed `run_arm()`
- `data.py`: explicit formatting, filtering, mixing, and token-dose helpers
- `evals.py`: versioned QA, LM, multiple-choice, and generation evaluators
- `influence.py`: paired baseline/treatment orchestration using `run_arm()`

Install the optional dependencies with:

```bash
pip install 'reasoning-core[training]'
```

An influence experiment is a baseline `ArmPlan` plus one or more treatment
plans. Dataset factories are called independently, and the same supplied model
state is restored before every arm:

```python
from reasoning_core.training.arm import ArmSpec
from reasoning_core.training.influence import ArmPlan, run_influence

baseline = ArmPlan(ArmSpec("study", "baseline", initialization_id="sha256:..."), main_data)
treatment = ArmPlan(
    ArmSpec("study", "task-x", aux_fraction=0.2, initialization_id="sha256:..."),
    mixed_data,
)
result = run_influence(model, tokenizer, initial_state, baseline, (treatment,), evaluate=evaluate)
print(result.deltas)
```

`ArmSpec.spec_id` covers the engine version and complete serialized spec. Status files additionally
record engine, package, dependency, initialization, data, and evaluation IDs.
Callers should provide immutable content or revision IDs for inputs that cannot
be inferred from streaming objects.

Ordering is never implicit: every `mix_streams()` call must choose a
`shuffle_buffer`. Use `0` to reproduce the legacy influence protocol. Enabling a
positive deterministic shuffle is a protocol change and produces a different
arm identity when recorded in `ArmSpec`.

The former `dev_*` modules remain compatibility imports. New code should import
the canonical modules above.
