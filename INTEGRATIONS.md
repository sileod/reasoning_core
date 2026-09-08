# Reasoning Core integrations

[Back to the README](README.md)

## Prime Intellect Environments Hub

Install the Reasoning Core environment and evaluate a model with Prime Intellect:

```python
#!pip install uv  # Install uv if needed.
!uv tool install prime --with openai -q
!uv tool run prime -- env install sileod/reasoning-core-env

import os

from openai import OpenAI
from verifiers import load_environment

env = load_environment("reasoning-core-env")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
results = env.evaluate(
    client=client,
    model="gpt-4.1-mini",
    num_examples=20,
    rollouts_per_example=1,
)
df = env.make_dataset(results).to_pandas()
```

## reasoning-gym

Reasoning Core tasks can be registered in reasoning-gym and mixed with its native tasks:

```python
import reasoning_core
import reasoning_gym
from reasoning_gym.composite import DatasetSpec

reasoning_core.register_to_reasoning_gym()

specs = [
    DatasetSpec(name="leg_counting", weight=1, config={}),  # reasoning-gym
    DatasetSpec(name="arithmetics", weight=1, config={}),  # Reasoning Core
]
dataset = reasoning_gym.create_dataset(
    "composite",
    size=10,
    seed=42,
    datasets=specs,
)
```

Reasoning-gym tasks can also be generated through Reasoning Core:

```python
from reasoning_core import get_task

task = get_task("reasoning_gym")
example = task.generate_example(level=1, rg_task="lcm")
```

Omit `rg_task` to sample a random reasoning-gym task.

## Source layout

Python adapters live in [reasoning_core/integrations/](reasoning_core/integrations/README.md).
Standalone OpenEnv and Prime Intellect distributions live in
[integrations/](integrations/README.md), outside the core wheel.
