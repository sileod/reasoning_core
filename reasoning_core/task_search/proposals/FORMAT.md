# SFT-first wave proposals

This directory is the durable memory of task ideation. Generated proposal waves go
under `archive/`; executable task-search plans remain the `wave*.yaml` files one level
up. Keeping the two formats separate prevents implementation constraints from becoming
the definition of a useful training task.

Create a wave with:

```bash
export NVIDIA_API_KEY=...
python -m reasoning_core.task_search propose sft-wave-1 --count 12
```

The default model is NVIDIA NIM's free `moonshotai/kimi-k3` endpoint, because it is a
large model on a free tier. Any OpenAI-compatible endpoint works through `--endpoint`,
`--model` and `--api-key-env`; pass `--reasoning-effort none` for endpoints that reject
unknown request fields. The proposer
makes one creative call and one independent critic call per round. It first rejects
exact name collisions locally, then asks the critic to compare every candidate with a
catalog assembled from:

- shipped `Task` and `DevTask` classes;
- the descriptions in `GALLERY.md`;
- every executable `task_search/plans/wave*.yaml` plan (retired plans in `plans/.legacy/` are not
  scanned: an idea that still matters lives on as a task or as a proposal);
- every prior proposal wave in `proposals/archive/`.

The catalog hash and model-call hashes are stored in the wave. A previous proposal is
therefore remembered even when it was never implemented. Novelty is checked in one batch:
Kimi receives every prior one-liner and every current proposal, retrieves at least three
neighbors per proposal, and also checks for duplicates within the current batch.

## Format

Top-level fields describe the ideation run, not an implementation run:

```yaml
format_version: 1
kind: sft_task_proposals
name: sft-wave-1
created_at: 2026-09-01T00:00:00+00:00
objective:
  training_stage: sft
  requested: 12
  accepted: 12
  complete: true
catalog:
  sha256: <hash of the ordered novelty catalog>
  entries: 245
  sources: {gallery: 65, plan: 110, proposal: 20, task: 70}
generation:
  provider: integrate.nvidia     # derived from the endpoint host
  model: moonshotai/kimi-k3
  endpoint: https://integrate.api.nvidia.com/v1/chat/completions
  seed: 0
  temperature: 1.0
  reasoning_effort: max
  calls: []                 # request/response hashes, never credentials
proposals: []
rejected: []
```

## Provenance

A wave whose proposals carry `novelty.source: legacy` never went past the novelty critic,
so the only account of why its ideas are worth implementing is who supplied them. Those
waves must carry a top-level block naming the origin:

```yaml
provenance:
  kind: external            # external | proposer | legacy_file
  name: ASTRA0              # the name the source gave the wave, not the file stem
  received: 2026-09-04
  source: battery supplied verbatim by the repository owner in a Claude Code session
```

`legacy_file` is a file already in the repository (`external` came from `WAVE1.md`),
`external` is a wave handed over from outside it (`manual_high_value_80` arrived as pull
request #52), and `proposer` is this pipeline's own output, which does not need the block
because its critic calls are already recorded in `generation.calls`. Validation requires
it, so an unreviewed wave cannot be archived without saying where it came from.

A proposal is a name and a coverage summary, and nothing else:

```yaml
- id: P001
  name: canonical_snake_case_name
  summary: >-
    one packed line: the distinct problem modes, the operations or input families they
    range over, and what the answer is
  novelty:
    verdict: novel
    nearest_neighbors:
      - id: gallery:graph_pathfinding
        relationship: adjacent
        overlap: both execute graph algorithms, but the state transition differs
      - id: task:graph_operations:GraphSuccessors
        relationship: different
        overlap: both query a graph-local result
      - id: proposal:external:P014
        relationship: different
        overlap: both walk a directed structure
    substantive_difference: why this is not a rename or parameter change
    scores: {novelty: 5, sft_value: 4, feasibility: 5, clarity: 4}
    reason: critic summary
```

The summary is the same object a shipped task carries in its class `summary`, so a
proposal, a catalog entry and a finished task all speak one language and a proposal can
be compared against the library without importing anything.

Everything the schema used to demand -- difficulty ladders, prompt contracts, answer
types, oracle libraries, worked demonstrations -- is gone on purpose. Difficulty scaling
is a library-wide convention, not a per-task invention, and the rest are decisions an
implementor makes better while looking at real generated instances than a proposer makes
on paper. Emitting them per proposal produced pages that read the same every time and
froze choices nobody had evidence for. Unknown keys are now a validation error, so a
proposer that keeps writing them is told rather than silently trimmed.

Validation checks the name is canonical snake case and the summary is one trimmed line of
40-240 characters that says what is generated and what is answered. It does not pretend to
prove task quality. The novelty critic rejects `duplicate` and `variant` ideas; human
review should still precede compilation into an executable plan.

## From proposals to a wave

```bash
python -m reasoning_core.task_search plan proposals/archive/external.yaml \
  --name wave8 --variants 2
```

One proposal becomes `--variants` independent trials with the identical instruction. The
runner derives each trial's seed from `sha256(base_seed:trial_id)`, so the draws differ in
sampling rather than in brief, and validation says which of them actually worked. That is
the point of keeping proposals thin: a one-line summary does not determine an
implementation, so the wave runs several and keeps what survives.

`--base-ref` is resolved to a concrete commit when the plan is written, and the plan name
must be a lowercase Python identifier because it becomes a package directory under
`reasoning_core/tasks/generated/`.

## external, the reference wave

```bash
python -m reasoning_core.task_search import-legacy
```

`external` is the 80 candidates from `WAVE1.md`, which came from outside this pipeline
and were committed as a file; the import makes no model calls, because their descriptions
were already one-line coverage specs. It exists to be beaten: a proposed wave and this one
go through identical plan generation, identical validation and identical scoring, so "the
proposer earns its calls" is a measurable claim rather than an assumption. Imported
proposals carry `novelty.source: legacy` and no scores, because no critic ever reviewed
them -- which is also why beating them is the easier half of the comparison.

It is not `wave0`. The plan `wave0.yaml` is a different, earlier thing whose tasks ship
under `reasoning_core/tasks/generated/wave0/`.

If bounded generation rounds produce fewer accepted ideas than requested, the archive is
still written with `objective.complete: false` and the command exits 2. This preserves
the paid model calls and rejected-idea audit instead of losing them on an exception.
