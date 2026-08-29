import random

random.seed(533370632)

from reasoning_core.tasks.generated.wave0.n02_modular_congruence_system.modular_congruence_system import (
    ModularCongruenceSystem,
)

out = []
task = ModularCongruenceSystem()
for level, count in ((0, 2), (2, 2), (5, 2)):
    task.config.set_level(level)
    out.append(f"## Level {level}")
    for i in range(count):
        ex = task.generate_example()
        out.append(f"### Example {i + 1}")
        out.append(task.render_prompt(ex.metadata))
        out.append(f"**Answer:** {ex.answer}")
        out.append("")

with open(
    "reasoning_core/tasks/generated/wave0/n02_modular_congruence_system/samples_N2.md",
    "w",
) as f:
    f.write("\n".join(out).rstrip() + "\n")

print("wrote samples_N2.md")
