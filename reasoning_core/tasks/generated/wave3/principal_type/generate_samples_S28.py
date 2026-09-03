import os
import random

from reasoning_core.tasks.generated.wave3.s28_principal_type.s28_principal_type import PrincipalType

random.seed(3213085537)

_HERE = os.path.dirname(os.path.abspath(__file__))

task = PrincipalType()

levels = [0, 2, 5]
lines = []
for level in levels:
    task.config.set_level(level)
    lines.append(f"# Level {level}")
    lines.append("")
    examples = [task.generate_example() for _ in range(2)]
    for ex in examples:
        lines.append("Prompt: " + task.render_prompt(ex.metadata).replace("\n", " "))
        lines.append("")
        lines.append("Answer: " + ex.answer)
        lines.append("")

with open(os.path.join(_HERE, "samples_S28.md"), "w") as f:
    f.write("\n".join(lines))
