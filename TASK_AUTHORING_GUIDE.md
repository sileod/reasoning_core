# Task Implementation Guide

## Goal
Creating data providing useful cognitive primitives for pre-training and serving as useful general agents.

Implement tasks that are:
- concise in code,
- solver-backed (use strong external libraries instead of re-implementing),
- distributionally broad (high structural variety),
- robustly scorable (`score_answer(generate().answer) == 1`).

## Core Contract (from `reasoning_core/template.py`)
Every task should provide:
- `Config` subclass with `update(self, c)`.
- `Task` subclass implementing:
  - `generate(self) -> Problem`
  - `prompt(self, metadata) -> str`
  - `score_answer(self, answer, entry) -> float | Reward` (or rely on default exact match)

`Problem` must include:
- `metadata` (dict/easydict),
- `answer` (ground-truth string),
- optional `metadata["cot"]`.

`Task.generate_example(...)` automatically adds metadata:
- `_task`, `_level`, `_config`, `_time`, `_prompt_tokens`, `_cot_tokens`.

## Config and Difficulty Scaling
Base `Config` protected fields:
- `c`: difficulty step size,
- `level`: current level,
- `seed`: RNG seed,
- `size`: optional dataset size.

Important behavior:
- Int-typed fields (except `level/size/seed`) are tracked internally as floats and stochastically rounded on read.

Design rules for `update(c)`:
- monotonic difficulty increase,
- no mutation of `seed`/`c`,
- keep generation solvable and diverse,
- avoid brittle jumps (prefer gradual increments).

## Reasoning-Core Philosophy
1. External libraries first:
- Use domain solvers/parsers/symbolic engines (`sympy`, planning engines, grammar libs, etc.).
- Do not hand-roll complex validators/solvers if a stable library exists.

2. Concise generation logic:
- Keep task code short and auditable.
- Push heavy correctness checks to proven toolchains.

3. High generality of distribution:
- Randomize structure, not just surface text.
- Avoid narrow templates that overfit lexical patterns.
- Prefer configurable families of instances over one fixed style.

4. Reward quality over strict formatting:
- Reward semantic correctness first, with optional light format penalties.
- Use `Reward(...)` tags when useful for diagnostics.

## Minimal Task Skeleton
```python
from dataclasses import dataclass
from reasoning_core.template import Task, Problem, Config, edict
from reasoning_core.utils import score_scalar

@dataclass
class MyTaskConfig(Config):
    n_vars: int = 2
    depth: int = 3

    def update(self, c=1):
        # used to scale difficulty
        self.n_vars += c
        self.depth += c

class MyTask(Task):
    def __init__(self, config=MyTaskConfig()):
        super().__init__(config=config)

    def generate(self):
        # Build instance using external libs when possible.
        metadata = edict({"instance": "...", "cot": "...optional..."})
        answer = "..."
        return Problem(metadata=metadata, answer=answer)

    def prompt(self, metadata):
        return f"Solve: {metadata['instance']}\nAnswer only."

    def score_answer(self, answer, entry):
        # answer is the answer to score (e.g. LLM prediction)
        # entry is a problem; entry.answer is the ground truth
        return score_scalar(answer, entry)  # or custom semantic checker
```

## Quality Checklist
- `task = MyTask(); x = task.generate_example()` works.
- `task.score_answer(x.answer, x) == 1`.
- Wrong/random answers do not all score `1`.
- `task.validate()` passes.
- `config.set_level(1)` changes difficulty, not `seed/c`.
- Prompt is unambiguous about output format.
- Metadata is sufficient for offline debugging (instance params, optional `cot` entry).

## Registration and Discovery
- Any `Task` subclass in `reasoning_core/tasks/*.py` is auto-discovered by AST and lazy-loaded through `reasoning_core.__init__.py`.
- `task_name` defaults to snake_case class name.
