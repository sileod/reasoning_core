import random
from pathlib import Path

from reasoning_core.tasks.generated.wave11.distinguishing_witness.distinguishing_witness import (
    DistinguishingWitness,
)

random.seed(2600584944)

out = Path(__file__).with_name("samples_P006v1.md")
task = DistinguishingWitness()

lines = []
lines.append("# Samples for distinguishing_witness (P006v1)")
lines.append("")

for lvl in (0, 2, 5):
    lines.append("## Level %d" % lvl)
    lines.append("")
    for i in range(2):
        task.config.set_level(lvl)
        ex = task.generate_example()
        lines.append("### Example %d (level %d)" % (i + 1, lvl))
        lines.append("")
        lines.append(task.render_prompt(ex.metadata))
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(ex.answer)
        lines.append("")

out.write_text("\n".join(lines))
print("wrote", out)
