# Task Implementation Guide

## Goal
Craft verifiable procedural data generator targetting speficic capabilities.
The data should be useful to learn cognitive primitives for language understanding and processing skill.
Data is intended designed for both pre-training (next token prediction) or post-training.
The data should provide high structural variety, but lexical/surface variety is not a priority.
This data should be used alongside natural data already providing surface variations.

Implement tasks that are:
- concise in code, easy to audit
- preferaby solver-backed (use strong external libraries instead of re-implementing),
- distributionally broad (high structural variety),
- focused on interesting cases without being templatic,
- not mostly solvable with shortcuts (some are good for robustness but they should be rare),
- verifiable, formal and robustly scorable (`task.score_answer(entry.answer, entry) == 1`).
- favour answer uniqueness if possible (e.g. specify lexicographic order) to ease next token prediction training.

## Core Contract (from `reasoning_core/template.py`)
Every task should provide:
- `Config` subclass with `apply_difficulty(self, level)`.
- `Task` subclass implementing:
  - `generate_entry(self) -> Entry`
  - `render_prompt(self, metadata) -> str`
  - `score_answer(self, answer, entry) -> float | Reward` (or rely on default exact match)

`Entry` must include:
- `metadata` (dict/easydict),
- `answer` (ground-truth string),

`Task.generate_example(...)` automatically adds metadata:
- `_task`, `_level`, `_config`, `_time`, `_prompt_tokens`, `_answer_tokens`,
- `_generator_name`, `_generator_version`, `_generator_commit`, `_task_version`,
- `_task_behavior_hash` (AST-based module hash; ignores whitespace, comments, and docstrings).
- `_deduplication_key` (128-bit hash of the canonical prompt/answer pair, with shallow
  payload order normalized; override `deduplication_key()` only for safe semantic invariances).

## Config and Difficulty Scaling
Base `Config` protected fields:
- `level`: current level,
- `seed`: RNG seed (do not use it. do not seed anything explictly unless it is requested.)

Important behavior:
- Int-typed fields (except `level/size/seed`) are tracked internally as floats and stochastically rounded on read.
- `set_level(level)` resets to the base config and applies difficulty from that base state.
- `apply_difficulty(level)` is the preferred explicit difficulty knob.
- Deprecated/legacy configs may still rely on `update(c)` through the base compatibility fallback; do not add `update(c)` to active task configs.

Design rules for `apply_difficulty(level)`:
- monotonic difficulty increase,
- no mutation of `level`,
- keep generation solvable and diverse
- update should change knobs (problem sizes or reasoning depth, etc)
- do not hardcode different subtasks (do not use "if level ... then ...")
- use direct formulas instead of recursively calling legacy update logic.

Use `Config_difficulty_knob_migration.md` and `assert_difficulty_update_equivalence(...)` when migrating existing configs.

Rough reference:
Level 0 should be as simple as possible while ensuring diversity (for example in a task where we generate graphs for shortest path prediction, 3 nodes are not enough because the combinatorics run out quickly)
Level 5 should be tough even for large LLMs.

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
- Reward semantic correctness first but prefer soft scoring.
- Blatantly inccorect answer should be reward 0.0, correct answer should have reward 1.0.
- Use `Reward(...)` tags when useful for diagnostics.

## Minimal Task Skeleton
```python
from dataclasses import dataclass
from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

@dataclass
class MyTaskConfig(Config):
    n_vars: int = 2
    depth: int = 3

    def apply_difficulty(self, level):
        # Use shared stochastic rounding for int-typed difficulty fields.
        self.n_vars = sround(self.n_vars + level)
        self.depth = sround(self.depth + level)

class MyTask(Task):
    # Do not put "Task" in the task name
    config_cls = MyTaskConfig

    def generate_entry(self):
        # Build instance using external libs when possible.
        metadata = edict({"equation": "...", "cot": "...optional..."})
        metadata.payload = {"equation": metadata.equation}
        answer = "..."
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        # Specify the answer format clearly, refer to it as "the answer" or "answer".
        # Do not use answer as a verb, do not use "return".
        # The wording logic should live here and not be buried in generation.
        return f"{render_payload(metadata.payload)}\n\nThe answer is a scalar."

    def score_answer(self, answer, entry):
        # Answer is the answer to score (e.g. LLM prediction)
        # entry is a problem; entry.answer is the ground truth
        # use ast.literal_eval for safety if evaluation is need
        # leniency is helpful (e.g. score 0.5 for half answer)
        # but 1 should be reserved for correct answers
        return score_scalar(answer, entry)  # or custom semantic checker
```

## Quality Checklist
- `task = MyTask(); x = task.generate_example()` works.
- `task.score_answer(x.answer, x) == 1`.
- Wrong/random answers do not all score `1`.
- `task.validate()` passes.
- `task.validate(cache=True)` may be used for local cached validation examples.
- `config.set_level(1)` changes difficulty.
- Prompt is unambiguous about output format.
- Prompt is as concise as possible while allowing meaningful zero-shot solvability.
- Answers should be short and canonical, valid SFT targets.
- Metadata is ideally sufficient for offline debugging (instance params, optional `cot` entry).
- If a task uses labeled prompt blocks, store them as a plain JSON-serializable
  `metadata.payload` mapping and render them with `render_payload(metadata.payload)`.
  Do not store renderer/helper objects in metadata.
- Metadata is not too large (should not blow up memory).


## Registration and Discovery
- Any `Task` subclass in `reasoning_core/tasks/*.py` is auto-discovered by AST and lazy-loaded through `reasoning_core.__init__.py`.
- `task_name` defaults to snake_case class name.

## Gallery
- If requested, refresh examples with `python scripts/build_gallery.py`.
- Gallery generation uses cached validation examples by default and builds missing cache entries.
- Use `--refresh-cache` to regenerate cached examples for the current task behavior hash/config.
- Use `--no-cache` to use balanced batch generation instead.
- Use `--taskrow-cache task_diagnostics/cache/task_rows/<cache_id>` to reuse diagnostics TaskRow examples before generating missing entries.
- The cache is keyed per task and level, and keeps only the latest record for each key.
