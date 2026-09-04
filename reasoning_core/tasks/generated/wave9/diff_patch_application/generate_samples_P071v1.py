import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.diff_patch_application.diff_patch_application import (
    DiffPatchApplication,
)

random.seed(911077872)

out = Path(__file__).with_name("samples_P071v1.md")

lines = ["# P071v1 samples: diff_patch_application", ""]

task = DiffPatchApplication()
for level in (0, 2, 5):
    lines.append(f"## Level {level}")
    lines.append("")
    task.config.set_level(level)
    for i in range(2):
        e = task.generate_example()
        prompt = task.render_prompt(e.metadata)
        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append(prompt)
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(e.answer)
        lines.append("")

with open(out, "w") as f:
    f.write("\n".join(lines))
