# Reasoning Core ◉


reasoning-core is a text-based RLVR for LLM reasoning training.
It is centered on expressive symbolic tasks, including full fledged FOL, formal mathematics with TPTP, formal planning with novel domains, and syntax tasks.

🤗 [https://hf.co/collections/reasoning-core/datasets](https://huggingface.co/collections/reasoning-core/datasets)

# Standalone
```python
pip install reasoning_core

from reasoning_core import list_tasks, get_task, score_answer

T = get_task('arithmetics')()
x = T.generate_example()
assert score_answer(x.answer, x)==1
```

# Parallel generation script
Run `bash run_generate.sh` for multi-threaded generation to json files (readable by Huggingface Datasets).


# Task examples and task authoring guide
[GALLERY](https://github.com/sileod/reasoning_core/blob/main/GALLERY.md) (names link to task code)  
[TASK_AUTHORING_GUIDE](https://github.com/sileod/reasoning_core/blob/main/TASK_AUTHORING_GUIDE.md)

# Integrations

### Prime Environment Hub
```python
#!pip install uv #install uv if needed
!uv tool install prime --with openai  -q
!uv tool run prime -- env install sileod/reasoning-core-env

from verifiers import load_environment
import os; from openai import OpenAI

env = load_environment("reasoning-core-env")

client = OpenAI( base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")) #🔑
results = env.evaluate(client=client, model="gpt-4.1-mini", num_examples=20, rollouts_per_example=1)
df=env.make_dataset(results).to_pandas()
```

### Reasoning gym integration

We use a custom interface but compatible interface. Our tasks, which are mostly orthogonal to RG, can be imported in it.
```python
import reasoning_gym, reasoning_core
from reasoning_gym.composite import DatasetSpec

reasoning_core.register_to_reasoning_gym() # registers RC tasks into RG

specs = [
    DatasetSpec(name='leg_counting', weight=1, config={}),  #from reasoning_gym 🏋
    DatasetSpec(name='arithmetics', weight=1, config={}),  #from reasoning_core ◉
]
D=reasoning_gym.create_dataset('composite', size=10, seed=42, datasets=specs)
```

And the other way around:
```python
frm reasoning_core import get_task
t=get_task('reasoning_gym')
t.generate_example(level=1, rg_task='lcm') #or unspecified for random task
```

## Citation
```
@article{lacombe2026reasoningcore,
  title={Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training},
  author={Lacombe, Valentin and Quesnel, Valentin and Sileo, Damien},
  journal={arXiv preprint arXiv:2603.02208},
  year={2026},
  url={https://arxiv.org/abs/2603.02208}
}
```
