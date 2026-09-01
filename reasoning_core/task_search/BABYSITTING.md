# Babysitting task-search runs

A babysitter observes and reports; the coordinator remains the only process that ends
generation, starts validation, assigns status, or promotes a candidate. While a trial is
active, do not edit its worktree, send signals, rerun validation, or remove worktree
metadata.

## Inspect a run

Read the invocation's `summary.json` first. A missing per-trial `run.json` means the
trial is still active or the coordinator stopped before recording it. Once `run.json`
exists, inspect:

- `status` and `outside_owned_path`;
- `task_metadata_matches` and `parent_source_id`;
- `sample_review`, `contract_audit`, and `validation`;
- `candidate` hashes and `resource_limits`;
- `steps`, especially whether the worker exhausted its budget.

Then read the bounded artifacts: `contract_audit.log`, `validation.log`, the generated
`samples_<ID>.md`, and the candidate diff. Avoid printing whole event logs because prompts
and tool outputs can be large.

For live progress, sample at most every 30–60 seconds. Count event types, tool calls, and
self-check results; a zero-length log during startup is normal. `trajectory.py` summarizes
these signals without changing the run:

```bash
python -m reasoning_core.task_search.trajectory /path/to/invocation
```

## Interpret status

Statuses use the first failed gate:

`timed_out → harness_failed → scope_violation → no_implementation → metadata_mismatch →
sample_review_failed → contract_failed → undiscoverable → candidate_mutated →
sample_not_reproducible → validation_failed → answers_impossible → success`

A later status implies the earlier gates passed. `orchestration_error` appears only in
the invocation summary and indicates that the coordinator itself failed before it could
write a trial record.

Classify a completed trial as:

- `review`: every automated gate passed; inspect semantics and samples before promotion;
- `retry`: a transient provider/harness failure or a narrow test-only mistake;
- `reject`: scope/provenance violations, broken scoring, invalid examples, or weak task
  semantics.

Recommendations do not launch retries. Start a new invocation explicitly so provenance
and attempts stay separate; runs are never resumed in place.

## Cleanup

Worktrees are retained intentionally. Remove them only after the wave has finished and
its candidates have been reviewed, using normal `git worktree remove` followed by
`git worktree prune` from the checkout. Never delete an active invocation directory.
