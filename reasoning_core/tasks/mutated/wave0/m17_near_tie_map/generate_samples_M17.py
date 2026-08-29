import random

from reasoning_core.tasks.mutated.wave0.m17_near_tie_map.near_tie_map import NearTieMap

random.seed(2846187700)

OUT = "reasoning_core/tasks/mutated/wave0/m17_near_tie_map/samples_M17.md"
LEVELS = [0, 2, 5]
PER_LEVEL = 2

lines = []
for level in LEVELS:
    task = NearTieMap()
    task.config.set_level(level)
    lines.append(f"# Level {level}\n")
    for i in range(PER_LEVEL):
        x = task.generate_entry()
        lines.append(f"## Example {i + 1}\n")
        lines.append("### Prompt\n")
        lines.append(x.metadata.english)
        lines.append("")
        lines.append("### Answer\n")
        lines.append(x.answer)
        lines.append("")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print("wrote", OUT)
