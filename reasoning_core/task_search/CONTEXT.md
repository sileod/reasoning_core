# Wave worker context

Implement one auditable reasoning task trial, not a general framework.

- Follow `TASK_AUTHORING_GUIDE.md` and use `GALLERY.md` for prompt/answer style.
- Keep canonical answers short and scoring exact or semantically robust.
- Prefer a trusted external solver and independently check generated instances.
- Make difficulty reflect structural reasoning, not only larger prompts.
- Keep one intended distributional change per mutation.
- For mutations, preserve the supplied parent source identity and `TASK_META`.
- Add focused tests beside the trial implementation inside the owned directory.
- Generate and inspect the required `samples_<trial-id>.md`; mechanical
  validation does not replace reviewing actual prompt/answer examples.
- Local `python -c` introspection is permitted inside the sandbox; do not prefix
  commands with `cd` and do not use it for network access.
- Do not modify shared registries; recursive discovery handles the trial module.
- Do not run influence experiments or promote the candidate during implementation.
