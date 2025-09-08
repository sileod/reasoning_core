# Reasoning core ◉


reasoning-core is a text-based RLVR for LLM reasoning training.
It is centered on expressive symbolic tasks, including full fledged FOL, formal mathematics with TPTP, formal planning with novel domains, and syntax tasks.


# Prime Environment Hub
```python
#!pip install uv #install uv if needed
!uv tool install prime --with openai  -q
!uv tool run prime -- env install sileod/reasoning-core-env

from verifiers import load_environment
import os; from openai import OpenAI

env = load_environment("reasoning-core-env")

os.environ["OPENROUTER_API_KEY"] = "" #✍️ write your key
client = OpenAI( base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
results = env.evaluate(client=client, model="gpt-4.1-mini", num_examples=20, rollouts_per_example=1)
df=env.make_dataset(results).to_pandas()
```

# Standalone
```python
pip install reasoning_core

from reasoning_core import list_tasks, get_task, score_answer

T = get_task('arith')()
x = T.generate_example()
assert score_answer(x.answer, x)==1
```

# Generation
Run `bash run_generate.sh` for multi-threaded generation to json files (readable by Huggingface Datasets).


# Reasoning gym

We use a custom interface, leaner than reasoning-gym (RG). But our tasks, which are all orthogonal to RG, can be imported in it.

```python
import reasoning_gym

from reasoning_core import register_to_reasoning_gym
register_to_reasoning_gym()

specs = [
    # here, leg_counting tasks will make up two thirds of tasks
    DatasetSpec(name='leg_counting', weight=2, config={}),  #from reasoning_gym 🏋
    DatasetSpec(name='arith', weight=2, config={}),  #from reasoning_core ◉
]
D=reasoning_gym.create_dataset('composite', size=10, seed=42, datasets=specs)

```