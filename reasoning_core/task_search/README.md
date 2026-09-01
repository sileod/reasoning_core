# Task Search

Reading this for a code review? Start with [REVIEW.md](REVIEW.md): where the
weight is, what has been measured, and the weak points already known.

`reasoning_core.task_search` distributes a versioned search plan to independent
coding-harness workers. It is provider- and harness-neutral: the selected
harness resolves model access from user-local configuration or Harness Link,
while this repository records only non-secret generation metadata.

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
directory and private OpenCode runtime directory. Host `/run` and `/tmp` are
hidden, PID/IPC/UTS/cgroup namespaces are isolated, and all capabilities are
dropped. The generated OpenCode profile
also denies subagents and web tools, and only permits `git status`, `git diff`,
and the trial's validation commands through the shell. After a worker exits, the
coordinator independently rejects changes outside the owned directory and runs
validation itself.

This enforces write confinement even for shell subprocesses and prevents access
to host daemon sockets such as Docker's. It is not a confidentiality or network
sandbox: workers can read ordinary host paths visible to the launcher, and the
OpenCode process must contact its configured model provider. Use a container or
VM for adversarial models or secret-bearing hosts.

## Commands

Task ideation is deliberately separate from executable waves. Generate an SFT-first
proposal archive with a novelty catalog and an independent Kimi K3 critic before asking
coding workers to implement anything:

```bash
export NVIDIA_API_KEY=...
python -m reasoning_core.task_search proposal-catalog
python -m reasoning_core.task_search propose sft-wave-1 --count 12
python -m reasoning_core.task_search check-proposals \
  reasoning_core/task_search/proposals/archive/sft-wave-1.yaml
```

The proposal format and novelty-memory rules are documented in
[`proposals/FORMAT.md`](proposals/FORMAT.md). Archived proposals are included in future
novelty checks even if they were never implemented. Proposal waves describe SFT learning
signal, distribution, curriculum, answer contract and shortcut risks; only reviewed
proposals should later be compiled into the verifier-oriented execution format below.

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

Harness Link can instead own only the provider adapter while task-search keeps
the prompt, permissions, provenance, worktree, sandbox, and validation:

```bash
curl -fsSL https://raw.githubusercontent.com/sileod/harness-link/main/install.sh | sh
export ALBERT_API_KEY=...
python -m reasoning_core.task_search run reasoning_core/task_search/wave0.yaml \
  --adapter harness-link --provider albert --model deepseek-v4-flash \
  --credential-env ALBERT_API_KEY \
  --jobs 1 --seed 20260828 --trial M1
```

The adapter and adapter version are recorded separately from the provider,
model, OpenCode version, and Bubblewrap version. Task-search supplies its
OpenCode settings through `OPENCODE_CONFIG_CONTENT`; Harness Link deep-merges
the provider configuration into them. Plans remain provider-independent.

OpenCode and mini-SWE-agent can run behind the same plan, prompt, detached
worktree, mount sandbox, resource limits, metadata, and independent validator.
Mini currently uses Harness Link so provider setup stays out of this repository:

```bash
uv tool install mini-swe-agent
python -m reasoning_core.task_search run reasoning_core/task_search/wave0.yaml \
  --harness mini --adapter harness-link --provider albert \
  --model deepseek-v4-flash --credential-env ALBERT_API_KEY \
  --jobs 1 --seed 20260828 --trial N1
```

For a matched comparison, run the same trial twice with the same base commit,
model, seed, step limit, and timeout, changing only `--harness`. Each invocation
gets a separate timestamped directory. OpenCode writes JSON events; Mini writes
`harness.log` plus `runtime/trajectory.json`.

Antigravity is available through the newer unified Harness Link frontend. It
uses the Google account already authenticated by `agy`; Harness Link provider
overrides such as Albert do not apply to AGY:

```bash
curl -fsSL https://raw.githubusercontent.com/sileod/harness-link/main/install.sh | sh
agy models  # verifies the existing login
python -m reasoning_core.task_search run reasoning_core/task_search/wave0.yaml \
  --harness agy --adapter direct --model gemini-3.7-flash-low \
  --jobs 1 --seed 20260828 --trial N1
```

The runner calls `hlink agy`, creates a fresh AGY project rooted at the detached
worktree, streams JSON events, and records both the `agy` and `hlink` versions.
AGY does not expose a step ceiling, seed, temperature, or top-p through this
surface, so the recorded step maximum is null and the outer wall-time remains
the hard generation limit. All independent gates are unchanged.

For a conservative unattended sequential run, first complete the `pilot`
queue, then launch the weekend queue with one worker:

```bash
TASK_SEARCH_MODEL=albert/deepseek-v4-flash \
  scripts/run_task_search_weekend.sh
```

Set `TASK_SEARCH_HARNESS=mini` to use Mini for the same queue.

For Harness Link, use an unqualified provider model and select its adapter. A
credential file is optional launcher plumbing and is never copied into the plan
or run record:

```bash
TASK_SEARCH_ADAPTER=harness-link TASK_SEARCH_PROVIDER=albert \
TASK_SEARCH_MODEL=deepseek-v4-flash \
TASK_SEARCH_KEY_ENV=ALBERT_API_KEY TASK_SEARCH_KEY_FILE=/private/key-file \
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

Workers are bounded to 56 agent steps and 30 minutes by default; both limits are
recorded and configurable with `--max-steps` and `--timeout-seconds`.

Explicitly retryable provider failures (for example an OpenCode API event with
HTTP 429 and `isRetryable: true`) are retried twice by default with 30- then
60-second backoff. Each failed attempt is retained beside the final trial as
`<trial>.attemptN-<reason>`, and `retry_history` records why it was repeated.
Signals still receive one immediate infrastructure retry. Gate failures and
slow trials that made progress are never retried automatically. Configure this
with `--transient-retries` and `--retry-backoff-seconds`.

`--pace` sets how hard the worker is told to hurry, independently of that budget:
`hurry` (the default) tells it not to explore and to start writing immediately,
`steady` allows a few orientation calls, and `deliberate` asks it to propose two
or three formulations and say what a lazy solver could exploit before writing any
code. The pace is recorded in generation metadata, so waves remain comparable and
the stance can be A/B-ed rather than assumed. Everything outside the pacing block
is identical across the three. After the
worker exits, the coordinator imports every owned `Task` subclass, runs its real
`validate()` contract, and samples 64 additional examples to reject gold-answer
failures or blank/whitespace/gibberish answers that incorrectly score as fully
correct. This contract audit and the trial-authored pytest suite both run inside
the same write sandbox as the worker.

Because that sandbox is writable — the sample generator has to write into it —
the coordinator hashes every file under the owned path once the contract audit
has certified it, and re-hashes after the model-authored generator, pytest suite
and prior audit have run. A candidate whose own validation code rewrote it is
rejected as `candidate_mutated`, and the accepted tree hash and per-file hashes
are recorded in `run.json` under `candidate`, so a result keeps a referent after
its worktree is removed. `undiscoverable` rejects a task the audit can import but
`reasoning_core._discover_tasks` would skip: a leading underscore on the module,
or any underscored parent directory.

Each worker must create `generate_samples_<trial-id>.py`, run it, and then read
the resulting `samples_<trial-id>.md`. The file
contains at least two actual prompt/answer pairs at levels 0, 2, and 5. The
coordinator may record whether it recognizes the sample command and subsequent
read in a harness event stream, but neither version-sensitive observation is a
hard gate. It requires the generator and sections to exist, then runs the
generator during independent validation and requires byte-identical
output from the recorded trial seed. This makes qualitative inspection part of
generation instead of a later human-only check.

Runs are stored outside the checkout by default, under a sibling directory named
`.reasoning_core-task-search/<wave>/<timestamp>/`. Each trial contains its
prompt, generated permission config, OpenCode JSON events, validation log,
`run.json`, and persistent worktree. The invocation directory also contains a
summary.

## Babysitting an unattended wave

The babysitter is an observer, not another implementor. It may read run
artifacts and report progress, but must not edit a worktree, send signals to a
worker, rerun validation inside an active trial, remove Git worktree metadata,
or promote a candidate. The coordinator remains the only process that decides
when generation ends and when validation starts.

For an invocation directory stored in `run_dir`, inspect it safely with:

```bash
cat "$run_dir/summary.json"
find "$run_dir" -mindepth 2 -maxdepth 2 -name run.json -print
wc -l -c "$run_dir"/*/events.jsonl
du -sh "$run_dir" "$run_dir"/*/runtime 2>/dev/null
```

An absent `run.json` means that trial is still running or the coordinator was
interrupted. `summary.json` is rewritten after every completed trial. Once a
trial has `run.json`, read these fields first:

- `status`: only `success` passed all automated gates;
- `outside_owned_path`: any entry is a hard scope rejection;
- `task_metadata_matches` and `parent_source_id`: provenance checks;
- `sample_review`: durable sample artifacts and optional harness observations;
- `contract_audit` and `validation`: independent exit codes;
- `resource_limits`: the limits actually applied, rather than merely requested.

Then inspect `contract_audit.log`, `validation.log`, `samples_<ID>.md`, and the
candidate diff. Classify the result as `review` (all gates passed), `retry`
(harness timeout/transient provider failure or a small test-only mistake), or
`reject` (scope/provenance violation, broken scoring contract, or poor task
semantics). A babysitter may recommend a retry, but should leave starting it to
the coordinator or operator so attempts remain explicit and reproducible.

For live progress, count JSON event types and tool calls rather than printing
whole events: prompts and tool outputs can be very large. Poll every 30–60
seconds at most. OpenCode writes `events.jsonl`; Mini writes `harness.log` and a
trajectory under its runtime directory. A zero-length log during startup is
normal. Avoid recursive searches through the worktree and runtime cache while
the worker is active.

Each trial directory holds `prompt.md`, a harness config and log,
`validation.log`, `contract_audit.log`, `run.json`, and `worktree/`. Statuses are
assigned by first failing gate, in this order: `timed_out`, `harness_failed`,
`scope_violation`, `no_implementation`, `metadata_mismatch`, `sample_review_failed`,
`contract_failed`, `undiscoverable`, `candidate_mutated`,
`sample_not_reproducible`, `validation_failed`, `success`.
A later status therefore implies every earlier gate passed. A trial that raised
inside the coordinator itself is recorded as `orchestration_error` in
`summary.json` only, with
`error_type` and `error` and no `run.json` and no worktree path; that is a bug
report about the runner, not about the candidate task.

Independent sample generation, contract audit, and each validation command have
a separate five-minute wall-clock limit, configurable with
`--validation-timeout-seconds`. A timeout is recorded as exit code 124 and the
queue continues. The enclosing systemd scope and Bubblewrap's parent-death
handling apply to these commands too.

`sample_not_reproducible` identifies a non-reproducible sample artifact: the
coordinator hashes `samples_<ID>.md`, re-runs the generator, and requires the
same bytes back. A generator seeded from the clock, or otherwise unseeded, fails
this gate with every validation command still exiting 0, so read
`sample_validation.reproducible` before blaming the tests.

Interrupting the coordinator is safe: completed trials keep their `run.json` and
the last `summary.json`, and the in-flight trial keeps its worktree without a
record. Runs are never resumed in place — a new invocation always gets a fresh
timestamped directory. Worktrees are kept on purpose and are never pruned, so an
unattended wave grows the runs root steadily; remove them with `git worktree
remove` and `git worktree prune` from the checkout only after the wave has
finished and its candidates have been reviewed.

Bubblewrap is required and checked before any worker is launched. OpenCode's own
path-glob edit permissions are deliberately not used as the security boundary;
their behavior varies across OpenCode releases. The exact OpenCode and
Bubblewrap versions are recorded in task metadata and run records.

For AGY, `hlink -y` only translates to AGY's native unattended approval flag;
it is not treated as containment. The same outer Bubblewrap namespace keeps the
checkout read-only except for the trial-owned directory, and the coordinator's
scope audit remains authoritative. `--new-project --add-dir` is necessary AGY
workspace plumbing, not a security exception. AGY currently has no per-run
switch equivalent to OpenCode's web/subagent denials, so those model-level tools
are not policy-equivalent even though their filesystem writes remain confined.
The shell helper file and background-task artifact directory that AGY normally
writes under its home are overlaid from the private trial runtime; credentials,
settings, conversations, and the real installation remain read-only.

The coordinator does not commit, merge, or promote results. Review successful
worktrees first; promotion remains an explicit human action.

When a user systemd manager is available, `--resource-limits auto` also puts
every worker and validation process in a scope capped at 8 GiB RAM, 512 tasks,
and 400% CPU. The weekend launcher uses `--resource-limits required` and fails
before launching if those controls are unavailable. Limits and whether they were
applied are recorded; use `--resource-limits none` only for an explicitly
unbounded run.

The fail-closed weekend launcher requires a working user systemd manager. Check
it before leaving a run unattended:

```bash
systemd-run --user --scope --quiet --collect true
```

If the user manager does not survive logout, enable lingering where permitted
(`loginctl enable-linger "$USER"`) or launch from a persistent login/session.
The runner aborts before creating a trial when the required scope is unavailable.

The worker harness necessarily receives its provider credential. Coordinator-run
sample generation, contract audits, and tests remove every variable named with
`--credential-env`; the weekend launcher forwards `TASK_SEARCH_KEY_ENV`
automatically. This reduces credential exposure in model-authored validation
code, but does not turn the worker itself into a confidentiality sandbox.

Harness Link Spawn is not currently a task-search sandbox backend: its local
sandbox mounts the whole workspace read-write. It becomes a suitable stronger
backend once it can mount the detached worktree read-only and overlay only the
trial-owned subdirectory read-write. Until then, use the hardened Bubblewrap
backend rather than weakening the folder invariant.
