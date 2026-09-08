import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.correction_aware_summary.correction_aware_summary import (
    CorrectionAwareSummary,
)

random.seed(3123569846)

out = Path(__file__).with_name("samples_P014v1.md")
task = CorrectionAwareSummary()

lines = []
for level in (0, 2, 5):
    lines.append(f"## Level {level}")
    for _ in range(2):
        task.config.set_level(level)
        e = task.generate_example()
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append(task.render_prompt(e.metadata))
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(e.answer)
    lines.append("")

out.write_text("\n".join(lines))
