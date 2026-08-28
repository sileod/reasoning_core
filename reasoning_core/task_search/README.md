# Task Search

`reasoning_core.task_search` distributes a versioned search plan to independent
OpenCode workers. It is intentionally provider-neutral: OpenCode resolves model
references and credentials from user-local configuration, while this repository
records only non-secret generation metadata.

Each trial receives:

- the same ordered global-context files;
- one compact trial instruction;
- one repository-relative directory it owns;
- explicit validation commands;
- an isolated Git worktree based on the same resolved commit.

New-task trials write to their eventual `tasks/generated/` destination inside
the isolated worktree; they do not affect normal discovery until that worktree
is reviewed and explicitly promoted. Descendant trials write to
`tasks/mutated/`, which remains excluded from default task listings even after
promotion.

Each worker runs in a Bubblewrap mount namespace. The complete filesystem and
worktree are read-only, with writable bind mounts only for the trial's owned
directory and private OpenCode runtime directory. The generated OpenCode profile
also denies subagents and web tools, and only permits `git status`, `git diff`,
and the trial's validation commands through the shell. After a worker exits, the
coordinator independently rejects changes outside the owned directory and runs
validation itself.

This enforces write confinement even for shell subprocesses. It is not a
confidentiality or network sandbox: workers can read the host paths visible to
the launcher, and the OpenCode process must contact its configured model
provider. Use a container or VM for adversarial models or secret-bearing hosts.

## Commands

Validate and inspect a plan without launching models:

```bash
python -m reasoning_core.task_search check reasoning_core/task_search/wave0.yaml
python -m reasoning_core.task_search render reasoning_core/task_search/wave0.yaml N1
```

Every trial is one agent-sized workload with its own OpenCode session, prompt,
worktree, writable folder, seed, samples, and validation record. Named,
possibly overlapping queues select those independent workloads: `pilot`, `new_p0`,
`new_later`, `mutations_p0`, `mutations_p1`, and `weekend_p0`. The pilot contains
one new task and one mutation. `weekend_p0` contains the proposed first batch
(N1–N6, M1–M10, and M13), leaving the costlier geometry trials for a later run.

Run selected trials with a locally configured OpenCode provider:

```bash
python -m reasoning_core.task_search run reasoning_core/task_search/wave0.yaml \
  --model albert/deepseek-v4-flash --jobs 4 --seed 20260828 \
  --queue new_p0
```

For a conservative unattended sequential run, first complete the `pilot`
queue, then launch the weekend queue with one worker:

```bash
TASK_SEARCH_MODEL=albert/deepseek-v4-flash \
  scripts/run_task_search_weekend.sh
```

Override `TASK_SEARCH_QUEUE`, `TASK_SEARCH_SEED`, or append ordinary runner
arguments as needed. A summary is written before launch and after every completed
trial, and orchestration errors are recorded without preventing the remaining
single-worker queue from running. Use `tmux`, `screen`, or the local batch system
if the invoking terminal may disconnect.

The same public command accepts OpenCode Zen or OpenRouter references:

```bash
--model opencode/<model-id>
--model openrouter/<provider/model-id>
```

Use exact identifiers reported by `opencode models`. Tokens, custom endpoints,
and provider installers do not belong in the plan or run records.

The base seed deterministically produces a distinct seed for each trial, so
parallel scheduling does not change trial configuration. Seeds are forwarded to
OpenCode by default and recorded alongside temperature, top-p, model, agent,
and harness version. Use `--no-forward-seed` for a provider that rejects seed;
even when accepted, remote agentic sampling should be treated as best-effort
rather than bit-for-bit deterministic.

Workers are bounded to 48 agent steps and 30 minutes by default; both limits are
recorded and configurable with `--max-steps` and `--timeout-seconds`. After the
worker exits, the coordinator imports every owned `Task` subclass, runs its real
`validate()` contract, and samples 64 additional examples to reject gold-answer
failures or blank/whitespace/gibberish answers that incorrectly score as fully
correct. This contract audit and the trial-authored pytest suite both run inside
the same write sandbox as the worker.

Each worker must create `generate_samples_<trial-id>.py`, run its explicitly
permitted command, and then read the resulting `samples_<trial-id>.md`. The file
contains at least two actual prompt/answer pairs at levels 0, 2, and 5. The
coordinator checks that the generator and sections exist and that OpenCode read
the sample artifact after its final edit. This makes qualitative inspection part
of generation instead of a later human-only check.

Runs are stored outside the checkout by default, under a sibling directory named
`.reasoning_core-task-search/<wave>/<timestamp>/`. Each trial contains its
prompt, generated permission config, OpenCode JSON events, validation log,
`run.json`, and persistent worktree. The invocation directory also contains a
summary.

Bubblewrap is required and checked before any worker is launched. OpenCode's own
path-glob edit permissions are deliberately not used as the security boundary;
their behavior varies across OpenCode releases. The exact OpenCode and
Bubblewrap versions are recorded in task metadata and run records.

The coordinator does not commit, merge, or promote results. Review successful
worktrees first; promotion remains an explicit human action.
