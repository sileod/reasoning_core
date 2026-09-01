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

The default model is NVIDIA NIM's free `moonshotai/kimi-k3` endpoint. The proposer
makes one creative call and one independent critic call per round. It first rejects
exact name collisions locally, then asks the critic to compare every candidate with a
catalog assembled from:

- shipped `Task` and `DevTask` classes;
- the descriptions in `GALLERY.md`;
- every executable `task_search/wave*.yaml` plan;
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
  provider: nvidia-nim
  model: moonshotai/kimi-k3
  endpoint: https://integrate.api.nvidia.com/v1/chat/completions
  seed: 0
  temperature: 1.0
  reasoning_effort: max
  calls: []                 # request/response hashes, never credentials
proposals: []
rejected: []
```

Each proposal is deliberately SFT-oriented:

```yaml
- id: P001
  name: canonical_snake_case_name
  family: graph
  semantic_signature: execute one precise cognitive operation and name its output
  learning:
    cognitive_operation: the operation a student must execute
    trained_behavior: what predicting the answer should reinforce
    transfer_targets: [two nearby abilities that should benefit]
  data:
    instance_family: how structurally varied instances are sampled
    structural_variation: [three independent structural axes]
    difficulty:
      level_0: the simplest diverse case
      progression: the structural quantity that grows
      level_5: the deep case, not merely a longer prompt
    prompt_contract: all information needed to determine one answer
    answer:
      type: integer          # boolean, integer, fraction, string, list, or tuple
      canonicalization: the exact canonical representation
    balancing: how answer priors and structural regimes stay broad
  oracle:
    method: the trusted solver or concise reference algorithm
    library: networkx        # null when no external solver is appropriate
    independent_check: a genuinely different correctness check
    invariants: [at least two domain or construction invariants]
  quality:
    why_sft: why these prompt/answer pairs provide useful gradient signal
    shortcut_risks: [at least two lazy strategies and how generation defeats them]
    novelty_claim: the substantive capability absent from the known catalog
  demonstration:
    prompt: one short, fully specified example
    answer: one canonical target
  novelty:
    verdict: novel
    nearest_neighbors:
      - id: gallery:graph_pathfinding
        relationship: adjacent
        overlap: both execute graph algorithms, but the state transition differs
      - id: plan:wave7:T006
        relationship: different
        overlap: both use repeated graph updates
      - id: task:graph_operations:GraphSuccessors
        relationship: different
        overlap: both query a graph-local result
    substantive_difference: why this is not a rename or parameter change
    scores: {novelty: 5, sft_value: 4, feasibility: 5, clarity: 4}
    reason: critic summary
```

Proposal validation checks shape, canonical answer types, snake-case names, non-empty
curricula, structural axes, invariants, shortcut analysis, and demonstrations. It does
not pretend to prove task quality. The novelty critic rejects `duplicate` and `variant`
ideas; human review should still precede compilation into an executable wave.

If bounded generation rounds produce fewer accepted ideas than requested, the archive is
still written with `objective.complete: false` and the command exits 2. This preserves
the paid model calls and rejected-idea audit instead of losing them on an exception.
