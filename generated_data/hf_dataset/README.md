# LLM-generated reasoning-core tasks

Generated 2026-07-30 using openrouter/deepseek/deepseek-v4-flash, synthetic-pool via `litlm`, following the same task pipelines (sandboxed execution, type-checking, acceptance criteria) as the procedural (mesopy) version of reasoning-core, for a direct, apples-to-apples comparison against it.

Every row passed the task's own real validation pipeline at generation time (not just a superficial format check), and was re-verified again at packaging time by re-running the task's own scorer against its own reference answer.

## Rows per task

| task | rows | levels |
|---|---|---|
| code_analysis | 615 | 0, 1, 2 |
| code_execution | 606 | 0, 1, 2 |
| code_input_deduction | 613 | 0, 1, 2 |
| code_iterations | 608 | 0, 1, 2 |
| code_repair | 620 | 0, 1, 2 |
| code_runnability | 622 | 0, 1, 2 |
| type_inhabitation | 1218 | 0, 1, 2 |

## Fields

- `task`: task name
- `level`: difficulty level (0-4)
- `prompt`: the rendered prompt
- `answer`: the reference answer
- `metadata`: task-specific structured fields backing the prompt/answer
- `call_id` / `source`: provenance back to the exact LLM generation call (`null`/`"synthetic-pool"` for type_inhabitation/code_repair rows sampled from the shared synthetic function pool rather than one LLM call per row)
