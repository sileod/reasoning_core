# Task Search

`reasoning_core.task_search` separates SFT task ideation from implementation.
Proposal waves describe the learning signal and data distribution; reviewed ideas are
then compiled into executable waves for isolated coding workers.

## Propose tasks

The proposer builds a durable novelty catalog from shipped task summaries, prior wave
plans, gallery entries, and archived proposals. It asks Kimi K3 for structured SFT ideas,
retrieves nearest neighbors for every candidate, and uses a second model pass to reject
semantic variants—not just exact name collisions.

```bash
export NVIDIA_API_KEY=...
python -m reasoning_core.task_search proposal-catalog
python -m reasoning_core.task_search propose sft-wave-1 --count 12
python -m reasoning_core.task_search check-proposals \
  reasoning_core/task_search/proposals/archive/sft-wave-1.yaml
```

See [`proposals/FORMAT.md`](proposals/FORMAT.md) for the SFT-first proposal schema and
novelty rules. Proposal archives are immutable novelty memory, including ideas that were
reviewed but never implemented.

## Execute a reviewed wave

Validate or render a plan without launching a model:

```bash
python -m reasoning_core.task_search check reasoning_core/task_search/wave0.yaml
python -m reasoning_core.task_search render reasoning_core/task_search/wave0.yaml N1
```

Run a selected trial or queue with a model already configured in the chosen harness:

```bash
python -m reasoning_core.task_search run reasoning_core/task_search/wave0.yaml \
  --model opencode/<model-id> --jobs 1 --trial N1
```

The execution plan is provider-independent. Provider credentials, endpoints, installers,
and machine-specific launch commands belong in a local ignored
`README.providers.private.md`, not in plans, prompts, or run records.

## Execution model

Each trial receives the ordered context files, one compact assignment, one owned task
directory, explicit validation commands, and an isolated worktree at the same base
commit. The worker may read the repository but can write only to its owned directory and
private runtime directory. The coordinator independently checks scope, provenance,
discovery, task validation, sample reproducibility, tests, answer diversity, and source
stability.

New tasks are written under `reasoning_core/tasks/generated/`; descendants of an existing
task live under `reasoning_core/tasks/mutated/` and remain hidden from default listings.
The coordinator never commits or promotes candidates. Successful worktrees still require
human review.

OpenCode, mini-SWE-agent, and AGY can share the plan and outer worktree sandbox. Harness
and adapter choices are recorded independently of the model so matched runs can change
one factor at a time. Use `python -m reasoning_core.task_search run --help` for the
available harness, adapter, resource, retry, and sampling controls.

## Outputs and safety

Runs default to a sibling `.reasoning_core-task-search/<wave>/<timestamp>/` directory.
Each trial records its prompt, harness events, validation logs, sample review, immutable
candidate hashes, and `run.json`; the invocation directory has an incrementally updated
`summary.json`.

Bubblewrap is required. It confines writes and hides common host runtime sockets, but it
is not a confidentiality or network sandbox: workers can read ordinary paths visible to
the launcher and must contact their configured provider. Use a container or VM for
adversarial models or secret-bearing hosts. Validation subprocesses receive a sanitized
environment with configured provider credentials removed.

For architecture and known limitations, see [`REVIEW.md`](REVIEW.md).
For safe unattended-run monitoring, see [`BABYSITTING.md`](BABYSITTING.md).
