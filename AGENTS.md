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
- Avoid traversing `reasoning_core/openenv`, `.venv`, checkpoints, and other generated environments when searching.
- Keep code concise, favor external libraries when possible.
- Influence runs need no data setup: batteries unpack `reasoning_core/resources/battery_legs.zip` into `data_cache/` on first load. Never rebuild a leg to "refresh" it — leg identity is the sha256 of its bytes, so a rebuild silently forks the battery ID.

## Repository map and local checks

- `registry.py`: discovery and scoring dispatch; `template.py`: task contract; `runtime.py`: execution/cache helpers.
- `training/`: evaluation batteries and paired influence; `task_search/`: proposal/implementation orchestration; `reports/`: reporting tools. All are under `reasoning_core/`.
- Read [docs/workflows.md](docs/workflows.md) for runnable recipes and install requirements. Find tasks without imports: `python -m reasoning_core catalog 'query' --all --json`.
- With `pytest` installed, core changes: `python -m pytest -q tests/test_task_discovery.py tests/test_score_answer_dispatch.py tests/test_template_validation.py tests/test_workflow_cli.py`.
- Training changes: `python -m pytest -q tests/test_training_runtime.py` (training extra required). Search changes: run the relevant `tests/test_task_search_*.py` files.
- Operational artifacts and private checkouts are not source dependencies; use tracked files for shared workflows.
