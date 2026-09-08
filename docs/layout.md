# Repository boundaries and migration

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

Importing `evaluation` or its group/composition helpers does not load training.
Metrics depend on Torch, while arm execution additionally needs the training
extra (TRL/Transformers). Runtime adapters import the external framework only at
its boundary. Operational reports remain private/outside the public package.

## Compatibility

The old `reasoning_core.training.*`, `zero_shot_eval`, `collect`,
`generation_worker`, and collection-adapter Python imports alias their canonical
modules. This preserves access to private helpers and monkeypatch behavior as
well as public classes. Existing `python -m reasoning_core.collect` and
`python -m reasoning_core.generation_worker` commands forward to the new modules.
`from reasoning_core import evaluate_model` remains supported.

Legacy `dev_*` modules and `train_arm` stay available in this migration. We do not
rename individual task modules or rebuild battery data. Manifests remain at their
existing `training/` paths for file-based consumers; battery functions now live in
`evaluation/battery.py`. Historical-manifest retirement and task lifecycle changes
need their own consumer migration. Default-valued group fields do not change old
arm identities. Existing engine/provenance identifiers intentionally retain their
old spelling.

`task_search` is available from a checkout/editable install or source distribution,
not from the ordinary wheel. Its installed dependencies and external harness
requirements are documented in its README. Independent applications retain their
internal package structure; update checkout paths to root `integrations/`.

Use a separate worktree for this migration. Do not repoint a shared editable
installation or move active run/cache directories. Other agents can continue on
`main`; the PR must be reviewed and merged before their checkout sees any changes.
