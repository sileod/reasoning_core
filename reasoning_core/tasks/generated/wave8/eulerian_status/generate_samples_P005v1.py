import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.eulerian_status.eulerian_status import EulerianStatus

random.seed(3825253341)

task = EulerianStatus()

out = Path(__file__).with_name("samples_P005v1.md")
lines = []
for level in (0, 2, 5):
    lines.append("## Level %d" % level)
    for i in range(2):
        e = task.generate_example(level=level)
        lines.append("### Example %d" % (i + 1))
        lines.append("**Prompt:**")
        lines.append("")
        lines.append(task.render_prompt(e.metadata))
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(e.answer)
        lines.append("")
out.write_text("\n".join(lines))
