# reasoning-core OpenReward environment

This folder contains an OpenReward-compatible ORS environment for `reasoning-core`.

## Files

- `server.py`: OpenReward environment server implementation.
- `requirements.txt`: Python dependencies for local/dev builds.
- `Dockerfile`: Container spec used by OpenReward deployment.

## Local run

```bash
cd reasoning_core/openreward/reasoning_core_env
pip install -r requirements.txt
python server.py
```

## Quick local smoke test

```python
from openreward import OpenReward

or_client = OpenReward()
env = or_client.environments.get(name="reasoningcore", base_url="http://localhost:8080")

print(env.list_splits())
print(env.list_tools())
print(env.list_tasks("train")[:1])
```

## Configuration

Set optional environment variables before launch:

- `RC_NUM_TRAIN` (default `500`)
- `RC_NUM_TEST` (default `50`)
- `RC_SEED` (default `0`)
- `RC_PASS_THRESHOLD` (default `0.9`)
- `RC_HF_DATASET` (default `reasoning-core/symbolic-reasoning-env`)
- `RC_HF_CONFIG` (optional dataset config name)
- `RC_DISABLE_HF_FALLBACK=1` to disable Hugging Face loading and use procedural fallback

The task order is deterministic for fixed values of these variables.

## Notes

- The environment exposes a single `answer` tool.
- The tool returns a human-readable result with a rounded reward (`reward=0.000` format).
- "Accepted" is intentionally lenient and defaults to `reward >= 0.9` (configurable via `RC_PASS_THRESHOLD`).
- The tool accepts either plain-text answers or XML-wrapped answers (`<answer>...</answer>`), matching common evaluator output formats.
- By default, tasks are loaded from Hugging Face dataset `reasoning-core/symbolic-reasoning-env` using its native `train`/`test` splits.
- If Hugging Face loading fails, the environment falls back to deterministic procedural task generation.
