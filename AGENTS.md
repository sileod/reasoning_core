# Agent Notes

- Start with `TASK_AUTHORING_GUIDE.md` before adding or changing tasks. It explains the expected `Task`/`DevTask` shape, canonical answers, scoring, validation, and dataset hygiene.
- Use `GALLERY.md` for concrete examples of prompt/answer style. Keep new tasks similarly short, unambiguous, and easy to score.
- Prefer `Task` only for stable core datasets. Use `DevTask` for deprecated datasets.
- `generate()` should return a `Problem`, not `None`. Let retries happen inside generation helpers or raise a clear `RuntimeError` after bounded attempts.
- Keep answers canonical and compact: booleans, numbers, or exact constrained strings. Avoid asking models to copy long bodies unless the task is explicitly about generation.
- For behavioral changes, bump `task_version`; if absent, start with `task_version = 2`.
- Do not touch unrelated dirty files. Check `git status --short` first; this repo often has work in progress.
- `rg` may be unavailable in this environment. Fall back to `find`, `grep`, and `sed`.
- Task files live in `reasoning_core/tasks/` (not top-level `tasks/`).
- Avoid traversing `reasoning_core/openenv`, `.venv`, checkpoints, and other generated environments when searching.
- Keep code concise, favor external libraries when possible.
- Influence runs need no data setup: batteries unpack `eval_data/battery_legs.zip` into `data_cache/` on first load. Never rebuild a leg to "refresh" it — leg identity is the sha256 of its bytes, so a rebuild silently forks the battery ID.
