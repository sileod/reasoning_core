import random
from pathlib import Path

random.seed(3008605303)

from reasoning_core.tasks.generated.wave8.left_coset_identification.left_coset_identification import (
    LeftCosetIdentification,
)

OUT = Path(__file__).with_name("samples_P022v1.md")
task = LeftCosetIdentification()

lines = []
for level in (0, 2, 5):
    lines.append(f"## Level {level}")
    task.config.set_level(level)
    for i in range(2):
        e = task.generate_example()
        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append("```")
        lines.append(task.render_prompt(e.metadata))
        lines.append("```")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append("```")
        lines.append(e.answer)
        lines.append("```")
        lines.append("")

OUT.write_text("\n".join(lines))
print("wrote", OUT)
