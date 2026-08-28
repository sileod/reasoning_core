# Task Mutation Storage Guide

This guide defines how generated task descendants are represented and how their
source ancestry is stored. Task behavior, scoring, and validation remain covered
by `TASK_AUTHORING_GUIDE.md`.

## Folder roles

- `reasoning_core/tasks/*.py`: stable, hand-authored core tasks.
- `reasoning_core/tasks/generated/`: promoted machine-generated tasks. They stay
  here after promotion rather than moving into the top-level task folder.
- `reasoning_core/tasks/mutated/`: experimental descendants. They are discovered,
  but `list_tasks()` hides them unless called with `include_mutated=True`.

## Source objects

`SourceStore` stores exact source text by SHA-256:

```text
.evolution/objects/<first two hex characters>/<sha256>.py
```

`put(source)` is idempotent, immutable, and safe for concurrent identical
writes. `get(source_id)` verifies the digest and returns the exact text. Object
identity is independent of filenames and Git history.

## Descendant metadata

Each generated or mutated task carries a concise literal module-level mapping.
Use `None` for `parent_source_id` only when a genuinely new task has no source
parent:

```python
TASK_META = {
    "parent_source_id": "<sha256>",
    "idea": "increase compositional distractor depth",
    "hypothesis": "H1",
    "changes": "sample two-hop distractors instead of one-hop distractors",
    "generation": {
        "provider_name": "<provider-id>",
        "model_name": "<exact-provider/model-id>",
        "harness_name": "opencode",
        "harness_version": "<exact-version>",
        "agent_name": "<agent-profile>",
        "settings": {},
    },
}
```

- `parent_source_id` identifies the exact source used as the parent.
- `idea` names the capability or variation being explored.
- `hypothesis` is a stable hypothesis ID defined by the task-search wave.
- `changes` describes the concrete implementation or distribution change.
- `generation` records the resolved, non-secret generation setup. The provider
  label, exact model ID, harness name/version, agent profile, and non-default
  settings belong here; endpoints, API keys, and credential-file paths do not.

Keep these values short. Store run IDs, measurements, and reports in experiment
outputs, keyed by task source IDs, rather than embedding results in task files.

## Snapshot invariant

Read and store the parent before it is changed, moved, or used to derive a child:

```python
from reasoning_core.source_store import SourceStore, snapshot_parent

parent_source, task_meta = snapshot_parent(
    "reasoning_core/tasks/example.py",
    idea="increase compositional distractor depth",
    hypothesis="H1",
    changes="sample two-hop distractors instead of one-hop distractors",
    generation={
        "provider_name": "openrouter",
        "model_name": "<exact-provider/model-id>",
        "harness_name": "opencode",
        "harness_version": "<exact-version>",
        "agent_name": "build",
        "settings": {"variant": "high"},
    },
    store=SourceStore(),
)
```

`snapshot_parent()` returns the exact text it stored plus the mapping to inject
as the child's `TASK_META`. Using the returned text as the parent representation
prevents the stored parent and the represented parent from diverging.

When a descendant later becomes a parent, snapshot its complete source in the
same way. This produces an ancestry chain without requiring a Git commit, and
every historical parent remains retrievable after edits or renames.

Generation metadata should be injected by the launcher from the resolved
invocation, not authored by the generating model. A public launcher may select
any OpenCode provider/model reference; provider credentials and custom endpoint
configuration remain user-local.
