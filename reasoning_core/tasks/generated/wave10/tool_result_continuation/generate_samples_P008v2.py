import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.tool_result_continuation.tool_result_continuation import (
    ToolResultContinuation,
)

random.seed(2955773006)
TASK = ToolResultContinuation()
LEVELS = {0: 2, 2: 2, 5: 2}

out = Path(__file__).with_name("samples_P008v2.md")
chunks = ["# Tool Result Continuation v2 samples", ""]

for level, count in LEVELS.items():
    chunks.append(f"## Level {level}")
    chunks.append("")
    cfg = None
    for i in range(count):
        entry = TASK.generate_example(level=level)
        chunks.append(f"### Example {i + 1}")
        chunks.append("")
        chunks.append("Prompt:")
        chunks.append("")
        chunks.append("```")
        chunks.append(TASK.render_prompt(entry.metadata))
        chunks.append("```")
        chunks.append("")
        chunks.append("Answer:")
        chunks.append("")
        chunks.append("```")
        chunks.append(entry.answer)
        chunks.append("```")
        chunks.append("")
    chunks.append("")

out.write_text("\n".join(chunks), encoding="utf-8")
print(f"wrote {out}")
