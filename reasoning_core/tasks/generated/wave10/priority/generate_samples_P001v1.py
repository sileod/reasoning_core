import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.instruction_priority.instruction_priority import (
    Priority)

random.seed(3427748574)

OUT = Path(__file__).with_name("samples_P001v1.md")
task = Priority()

lines = []
lines.append("# Samples: instruction_priority (P001v1)\n")
for level in (0, 2, 5):
    lines.append(f"## Level {level}\n")
    for _ in range(2):
        task.config.set_level(level)
        x = task.generate_example()
        prompt = task.render_prompt(x.metadata)
        lines.append(f"**Prompt:**\n\n{prompt}\n")
        lines.append(f"**Answer:** {x.answer}\n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT)
