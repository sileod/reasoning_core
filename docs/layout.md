# Repository layout

| Location | Responsibility |
|---|---|
| `reasoning_core/tasks/`, core modules | Procedural generators, discovery, scoring contract |
| `reasoning_core/evaluation/` | Zero-shot evaluation, metrics, batteries, intrinsic rewards, influence, composition |
| `reasoning_core/evaluation/training/` | Data mixing, arm execution, optimization, checkpoints |
| `reasoning_core/generation/` | Generation workers and collection |
| `reasoning_core/integrations/` | Optional Python adapters |
| root `integrations/` | Independently packaged OpenEnv and Prime Intellect applications |
| `reasoning_core/task_search/` | Repository-only developer orchestration, excluded from the wheel |
| `scripts/` | Commands, smoke runs, and builders for new evaluation datasets |
| `docs/`, `docs/results/` | Protocols and maintained published results |

There is one import path per module. Reach for a module where the table says it
lives; nothing aliases anything else.

## Import cost

Importing `evaluation` or its group/composition helpers does not load training.
Metrics depend on Torch, while arm execution additionally needs the training
extra (TRL/Transformers). Runtime adapters import the external framework only at
its boundary, so core stays importable without `reasoning_gym` installed.

`reasoning_core/__init__.py` binds only the intended API. Registry discovery
internals live in `reasoning_core.registry` and are imported from there.

## Deliberate exceptions

Two things sit where the table would not predict, and both are load-bearing:

- **Battery manifests stay under `reasoning_core/training/`.** Running jobs pass
  `--eval-manifest reasoning_core/training/copyfree_battery_v8_tiny.json` and shipped arm
  specs record that path, so the files are addressed by path by consumers we do not
  control. The battery *functions* live in `reasoning_core/evaluation/battery.py`; only the JSON stayed.
- **Engine and provenance identifiers keep their existing spelling.** They are hashed into
  arm identities, so renaming them would silently split old and new results.

Group fields default to values that spell themselves absent, so adding them left every
pre-existing arm identity byte-identical. Any change here must preserve that: an ArmSpec
must keep hashing to the same `spec_id` across a refactor, or previously measured arms stop
pooling with new ones.

`task_search` is available from a checkout, editable install, or source distribution, not
from the ordinary wheel; its dependencies are documented in its README. The independent
applications under root `integrations/` retain their own package structure and pyprojects.

Operational reports, results and G5K orchestration are private and live outside this
repository.
