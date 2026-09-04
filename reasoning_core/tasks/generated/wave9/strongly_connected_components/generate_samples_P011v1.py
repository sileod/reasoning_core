import random
from pathlib import Path

random.seed(2305351643)

from reasoning_core.tasks.generated.wave9.strongly_connected_components.strongly_connected_components import (
    StronglyConnectedComponents,
)

out = Path(__file__).with_name("samples_P011v1.md")
task = StronglyConnectedComponents()

lines = []
for level in [0, 2, 5]:
    lines.append(f"# Level {level}")
    lines.append("")
    task.config.set_level(level)
    for _ in range(2):
        ex = task.generate_example()
        lines.append(task.render_prompt(ex.metadata))
        lines.append("")
        lines.append(f"Answer: {ex.answer}")
        lines.append("")

out.write_text("\n".join(lines))
