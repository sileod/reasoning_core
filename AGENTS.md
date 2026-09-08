# Agent Notes

- Start with `TASK_AUTHORING_GUIDE.md` before adding or changing tasks. It explains the expected `Task`/`DevTask` shape, canonical answers, scoring, validation, and dataset hygiene.
- Use `GALLERY.md` for concrete examples of prompt/answer style. Keep new tasks similarly short, unambiguous, and easy to score.
- Prefer `Task` only for stable core datasets. Use `DevTask` for deprecated datasets.
- `generate_entry()` should return an `Entry` (`generate()`/`Problem` are legacy aliases), not `None`. Let retries happen inside generation helpers or raise a clear `RuntimeError` after bounded attempts.
- Keep answers canonical and compact: booleans, numbers, or exact constrained strings. Avoid asking models to copy long bodies unless the task is explicitly about generation.
- For behavioral changes, bump `task_version`; if absent, start with `task_version = 2`.
- Do not touch unrelated dirty files. Check `git status --short` first; this repo often has work in progress.
- `rg` may be unavailable in this environment. Fall back to `find`, `grep`, and `sed`.
- Task files live in `reasoning_core/tasks/` (not top-level `tasks/`).
- Avoid traversing `integrations/openenv`, `.venv`, checkpoints, and other generated environments when searching.
- Keep code concise, favor external libraries when possible.
- Influence runs need no data setup: batteries unpack `reasoning_core/resources/battery_legs.zip` into `data_cache/` on first load. Never rebuild a leg to "refresh" it — leg identity is the sha256 of its bytes, so a rebuild silently forks the battery ID.

## Repository map and local checks

- `registry.py`: discovery and scoring dispatch; `template.py`: task contract; `runtime.py`: execution/cache helpers.
- Under `reasoning_core/`: `evaluation/` owns zero-shot, intrinsic reward, batteries, and influence; `evaluation/training/` owns the arm engine; `generation/` owns generation/collection; `integrations/` owns optional adapters; `task_search/` owns developer orchestration.
- There is one import path per module and nothing aliases anything else; if you move a module, repoint its callers rather than leaving an alias behind. Do not change run/battery identities when relocating: an `ArmSpec` must keep its `spec_id` and a battery its `identifier`.
- Battery manifests are data, under `reasoning_core/resources/batteries/`. A manifest pins its own `name`, which is hashed into the battery identifier recorded by measured arms, so add manifests freely but never rename or edit one.
- Standalone integration distributions live in root `integrations/`; reports and private ops remain outside the public package.
- Read [docs/workflows.md](docs/workflows.md) for runnable recipes and install requirements. Find tasks without imports: `python -m reasoning_core catalog 'query' --all --json`.
- `tests/` mirrors the package: `tests/tasks/`, `tests/evaluation/`, `tests/generation/`, `tests/integrations/`, `tests/task_search/`, `tests/scripts/`, with the registry and task-contract tests at the top level. Run the directory matching what you changed.
- With `pytest` installed, core changes: `python -m pytest -q tests/*.py`. Training changes: `python -m pytest -q tests/evaluation` (training extra required). Search changes: `python -m pytest -q tests/task_search`.
- Operational artifacts and private checkouts are not source dependencies; use tracked files for shared workflows.
