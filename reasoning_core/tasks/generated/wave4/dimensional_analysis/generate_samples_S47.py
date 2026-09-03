import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s47_dimensional_analysis.s47_dimensional_analysis import (
    DimensionalAnalysisTask,
)

random.seed(1833805255)

out = Path(__file__).with_name("samples_S47.md")


def render_level(task, level, n):
    task.config.set_level(level)
    lines = ["## Level %d" % level]
    for i in range(n):
        e = task.generate_entry()
        lines.append("### Example %d" % (i + 1))
        lines.append("**Prompt:**")
        lines.append("```")
        lines.append(task.render_prompt(e.metadata))
        lines.append("```")
        lines.append("**Answer:**")
        lines.append("```")
        lines.append(e.answer)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


task = DimensionalAnalysisTask()
sections = []
for level, n in [(0, 2), (2, 2), (5, 2)]:
    sections.append(render_level(task, level, n))

with open(out, "w") as f:
    f.write("# Samples S47\n\n")
    f.write("\n".join(sections))
