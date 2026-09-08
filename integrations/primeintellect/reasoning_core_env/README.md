# reasoning-core-env

### Overview
- **Environment ID**: `reasoning-core-env`
- **Short description**: Single-turn evaluation over `reasoning_core` procedural tasks with XML formatting.
- **Tags**: reasoning, procedural, single-turn, xml, synthetic

### Datasets
- **Primary dataset(s)**: Generated via [reasoning_core](https://github.com/sileod/reasoning_core)
- **Source links**: `reasoning_core` library
- **Split sizes**: Configurable counts for train/eval via loader args

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["think","answer"])`
- **Rubric overview**: Score computed via `reasoning_core` task-specific scorer; optional format component

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval reasoning-core-env
```

Configure model and sampling:

```bash
uv run vf-eval reasoning-core-env   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

