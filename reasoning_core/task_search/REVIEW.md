# Notes for a reviewer

Short orientation for someone reading `reasoning_core/task_search/` cold. The
README describes what the system does; this file says where the weight is, what
has actually been measured, and which parts I already think are weak.

## What the thing is

A coordinator that hands one reasoning-task-generator specification to an
independent coding agent, gives it a detached Git worktree and a write-confined
sandbox, and then re-validates the result itself. The agent's own claims are
never trusted: after it exits, the coordinator imports the task, runs its
`validate()` contract, generates 64 fresh examples, and re-runs the agent's
sample generator to check the bytes come back identical.

The premise being tested is that a cheap fast model can one-shot a decent
generator if the prompt and the gates are right, so nearly every fix goes into
the prompt or the gates rather than into more agent steps.

## Where to look, in order

| file | lines | what it is |
|---|---|---|
| `runner.py` | 1400 | the whole coordinator: plan loading, worktree/sandbox setup, prompt rendering, gates |
| `selfcheck.py` | 244 | the same gates, re-implemented to run *inside* the agent's sandbox in ~33s |
| `wave0.yaml` | 540 | the plan: one entry per trial, task spec + owned directory + queue |
| `prior_audit.py` | 83 | the gameability gate: can a constant answer beat the task? |
| `WAVE0.md` | 281 | per-trial specifications the prompt quotes from |

`runner.py` is one module and should probably not be. The parts I would split
are the plan model, the sandbox construction, and the gate pipeline; they have
no shared state beyond a trial record.

## The three non-obvious design decisions

**1. The gates are duplicated on purpose.** `runner.py` owns the authoritative
gates and `selfcheck.py` re-implements them for the agent to run on itself. The
agent otherwise cannot see why it failed — it exits, and only then does the
coordinator judge it. The contract audit itself cannot drift: `selfcheck`
`ast`-extracts `_CONTRACT_AUDIT` out of `runner.py` and runs that exact string.
The *pipelines* around it still differ, and that is a defect rather than a
design decision — `selfcheck` has discovery and 0/2/5 smoke gates the
coordinator did not have, so the prompt's claim that eleven PASSes are what the
harness scores was an overclaim. The coordinator now has a `discoverable` gate
too; the smoke gate is still selfcheck-only. The right end state is one
`gates.py` returning structured results that both callers use.

**2. The gate order is a precedence chain, not a set.** `runner.py` assigns the
first failing status: `timed_out → harness_failed → scope_violation →
no_implementation → metadata_mismatch → sample_review_failed → contract_failed →
undiscoverable → candidate_mutated → sample_not_reproducible → validation_failed
→ success`. A later status implies every earlier gate passed, which is what
makes the statuses comparable across waves. It also means a status names the
*first* thing wrong, never the cause — several waves were misread because of
this.

**3. The prompt is rendered per trial, inside `_run_trial`.** Editing the prompt
while a wave is running gives different trials different prompts and destroys
the comparison. The template lives in `runner.py`, which a running coordinator
has already imported, so editing that is safe mid-wave; the `context_files`
named in the plan are not, because `render_prompt` reads them from the live
checkout at `repo_root / relative` rather than from `base_commit`. Editing a
guide during a wave changes later workers. Resolving every prompt once before
the executor starts would fix this properly.

The `render` subcommand does **not** print the prompt a worker receives: it
calls `render_prompt(..., task_meta=None)`, while execution builds a model-,
seed- and budget-dependent `TASK_META` and passes it in. It is a template
preview; the prompt a worker actually got is `prompt.md` in its trial directory.

## What has actually been measured

- **Step ceiling.** 18 of 22 trials in an early wave died at exactly 55 tool
  calls, in both arms. Verification, not implementation, was eating the budget:
  agents re-derived the same checks with ad-hoc `python -c` calls. Consolidating
  every check into one self-check command and dropping the budget to 28 steps
  raised the success rate rather than lowering it.
- **Replay.** A wave of 18 trials that had *all* failed once closed at 10/18
  after the self-check landed, against 8/26 on the previous pass. This is a
  replay of failures only — regression to the mean moves it upward on its own
  and no trial could regress — so read the count, not a p-value. A clean paired
  A/B needs the whole set replayed, successes included; that is not done yet.
- **The 300-second clock is invisible.** Every validation command, the contract
  audit included, is killed at 300s. The audit generates 64 examples at the
  *default* config, so a generator averaging more than ~4.5s an example loses
  the trial on a clock nothing reports — the trial sees exit 124 and no
  explanation. Generation cost is heavy-tailed: one task measured 0.18s an
  example over a smoke run of 9 and 7.1s over 8 real ones, with a single
  instance at 33s. Extrapolating from cheap instances underestimated by ~25x.
  Hence selfcheck's `speed` gate times examples the way the audit makes them and
  quotes the worst as well as the mean.
- **`PYTHONHASHSEED` pins strings only.** Object-keyed dicts and sets iterate in
  `id()` order whatever the salt, so a three-process determinism check passed a
  non-deterministic generator about a quarter of the time. It now runs five
  processes at salts `0,0,1,1,2`, which separates "unstable across salts" from
  "unstable at a fixed salt".
- **Gameability.** Two of the first six new tasks were winnable without reading
  the prompt. `prior_audit.py` gates every trial on a constant-guess score above
  0.4. Three tasks are quarantined under `tasks/generated/_gameable/` for
  failing it.

## Weak points I already know about

- `runner.py` is a single 1400-line module (above).
- The two gate pipelines still differ around the smoke gate (above).
- `_sample_review` checks only that the four substrings `level 0`, `level 2`,
  `level 5` and `answer` appear *somewhere* in the samples file, while the prompt
  asks for two complete prompt/answer pairs at each level. The markers are now
  re-checked after the deterministic replay — previously they were read off
  whatever the worker left behind, so a stale file with the right markers passed
  and the replay could then overwrite it with anything — but nothing counts
  examples or checks they are non-empty.
- The gameability gate structurally excludes balanced binary tasks, and that is
  deliberate: `prior_audit` scores the most common answer against every sample,
  so a two-label task necessarily lands at or above 0.5 against a 0.4 threshold.
  A task whose reward is half free is not measuring reasoning. The real defect is
  that `wave0.yaml` contains trials specifying boolean answers while the global
  prompt tells workers to convert yes/no into witnesses — the plan contradicts
  the gate. Fix the plan, not the threshold.
- `report.py` is legacy: its default glob is `runs/ts_*/WAVE0/...` while runs now
  default to a sibling `.reasoning_core-task-search/...`, and it parses OpenCode
  events only, not Mini trajectories.
- `check` validates the plan syntactically — relative paths, overlap, IDs,
  queues — but does not resolve the base commit, verify context files exist at
  it, or confirm `owned_path` is under `reasoning_core/tasks`, which
  `_task_classes` later assumes. A plan can pass `check` and fail at launch.
- **The sandbox is write-confinement, not confidentiality.** Bubblewrap makes
  everything read-only except the trial's own directory, and the coordinator
  independently rejects changes outside it. But the agent can *read* ordinary
  host paths, and the shell allowlist is all that keeps it away from them — and
  that allowlist is not tight: every entry carries a trailing `*` (added in
  fb0357a, because exact-string matching was eating ~30% of the step budget) and
  `python -c *` is allowed outright, which is general-purpose. The `read` deny on
  `*.env` is a tool-level rule, not a filesystem one. That is consistent with the
  stated threat model — the model is not adversarial and Bubblewrap is the write
  boundary — but it means the shell policy protects nothing on its own. A real
  confidentiality boundary would be a container or VM.
- Worktrees are never pruned, on purpose, so an unattended wave grows the runs
  root steadily. Someone has to clean up.
- The plan is YAML with no schema; `check` validates it, but by hand.
- `sample_not_reproducible` is a good gate that reports badly: every validation
  command still exits 0 when it fires, so the log looks clean.

## The largest gap: nothing here checks whether the task is any good

Every gate is mechanical. Discovery, metadata, runtime, determinism, pytest, the
scoring contract, the constant-answer prior — a task can pass all of them and
still be a bad task. Nothing asks whether the difficulty axis is meaningful,
whether level 5 requires deeper reasoning than level 0, whether the task
duplicates one that already exists, or whether a mutation changed only the
variable its hypothesis names. `prior_audit` is the single semantic gate and it
tests exactly one failure mode.

This interacts badly with the step budget. The cheapest action available to a
worker that already has code is `selfcheck → patch the failing gate → selfcheck`,
so more steps buy convergence on the verifier rather than a better task, which is
consistent with the measurement above: cutting the budget from 56 steps to 28
raised the success rate. The plan compounds it — each trial arrives with `idea`,
`changes` and an `instruction` already fixed, so the worker is implementing an
assignment, not searching over formulations, and the prompt tells it to start
writing immediately and not explore.

Read the success rates accordingly. They measure *"implements a specified task
without tripping a mechanical gate"*, which is what the machinery is actually
built for, and not *"finds a good task"*. Separating ideation from
implementation — several short independent proposal workers with no write access,
a critic that ranks them, and only the winner entering this pipeline — is the
change most likely to matter, and it is not built.

## Running it without a model

```bash
python -m reasoning_core.task_search check  reasoning_core/task_search/wave0.yaml
python -m reasoning_core.task_search render reasoning_core/task_search/wave0.yaml N1
```

`render` prints the prompt *template* — see the note above. The exact prompt a
worker received is `prompt.md` in its trial directory.

## What rode along with this branch

Ten wave-0 tasks that passed every gate are promoted under
`tasks/generated/wave0/` (3) and `tasks/mutated/wave0/` (7). Four are
quarantined in `_gameable/` and `_nondeterministic/` and are excluded from
discovery; they are kept as regression material for the gates, not as tasks.
