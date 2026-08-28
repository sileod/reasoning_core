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

Each generated or mutated descendant carries a concise literal module-level
mapping:

```python
TASK_META = {
    "parent_source_id": "<sha256>",
    "idea": "increase compositional distractor depth",
    "hypothesis": "deeper distractors reduce shortcut accuracy at levels 3+",
    "changes": "sample two-hop distractors instead of one-hop distractors",
}
```

- `parent_source_id` identifies the exact source used as the parent.
- `idea` names the capability or variation being explored.
- `hypothesis` states the expected measurable effect. This is the field for
  grouping and testing task hypotheses across runs.
- `changes` describes the concrete implementation or distribution change.

Keep these values short. Store run IDs, measurements, and reports in experiment
outputs, keyed by task source IDs, rather than embedding results in task files.

## Snapshot invariant

Read and store the parent before it is changed, moved, or used to derive a child:

```python
from reasoning_core.source_store import SourceStore, snapshot_parent

parent_source, task_meta = snapshot_parent(
    "reasoning_core/tasks/example.py",
    idea="increase compositional distractor depth",
    hypothesis="deeper distractors reduce shortcut accuracy at levels 3+",
    changes="sample two-hop distractors instead of one-hop distractors",
    store=SourceStore(),
)
```

`snapshot_parent()` returns the exact text it stored plus the mapping to inject
as the child's `TASK_META`. Using the returned text as the parent representation
prevents the stored parent and the represented parent from diverging.

When a descendant later becomes a parent, snapshot its complete source in the
same way. This produces an ancestry chain without requiring a Git commit, and
every historical parent remains retrievable after edits or renames.
