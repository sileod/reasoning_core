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
coordinator judge it. Every gate the coordinator applies now has a self-check
counterpart, and the prompt tells the agent not to end its turn until all
eleven print PASS. **This duplication can drift**, and that is the first thing
worth criticising: `_CONTRACT_AUDIT` (a raw-string constant in `runner.py`) and
selfcheck's `contract` gate must stay in agreement, and nothing enforces it.

**2. The gate order is a precedence chain, not a set.** `runner.py:1121`
assigns the first failing status:
`timed_out → harness_failed → scope_violation → no_implementation →
metadata_mismatch → sample_review_failed → contract_failed →
sample_not_reproducible → validation_failed → success`. A later status implies
every earlier gate passed, which is what makes the statuses comparable across
waves. It also means a status names the *first* thing wrong, never the cause —
several waves were misread because of this.

**3. The prompt is rendered per trial, inside `_run_trial`.** Editing the
prompt while a wave is running gives different trials different prompts and
destroys the comparison. Prompt changes are held until a wave closes. If you
see a `PARKED` note or an unusually stale-looking prompt string, that is why.

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
- The gate duplication has no consistency test (above).
- **The sandbox is write-confinement, not confidentiality.** Bubblewrap makes
  everything read-only except the trial's own directory, and the coordinator
  independently rejects changes outside it. But the agent can *read* ordinary
  host paths, and the shell allowlist is the only thing keeping it away from
  them. That allowlist is matched on the **whole command string**, so it is
  brittle in a way agents trip over — appending `| tail -20` or `2>&1` to an
  allowed command makes it denied — and it must never grow a general
  content-reading command. Widening it to fix agent friction would be the wrong
  fix. A real confidentiality boundary would be a container or VM.
- Worktrees are never pruned, on purpose, so an unattended wave grows the runs
  root steadily. Someone has to clean up.
- The plan is YAML with no schema; `check` validates it, but by hand.
- `sample_not_reproducible` is a good gate that reports badly: every validation
  command still exits 0 when it fires, so the log looks clean.

## Running it without a model

```bash
python -m reasoning_core.task_search check  reasoning_core/task_search/wave0.yaml
python -m reasoning_core.task_search render reasoning_core/task_search/wave0.yaml N1
```

`render` prints the exact prompt a trial receives, which is the most useful
single artifact for reviewing the prompt-side decisions.

## What rode along with this branch

Ten wave-0 tasks that passed every gate are promoted under
`tasks/generated/wave0/` (3) and `tasks/mutated/wave0/` (7). Four are
quarantined in `_gameable/` and `_nondeterministic/` and are excluded from
discovery; they are kept as regression material for the gates, not as tasks.
