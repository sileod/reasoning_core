import os
import random

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
random.seed(59956088)

from reasoning_core.tasks.mutated.wave0.m16_defeasible_conflict_depth.m16_defeasible_conflict_depth import (
    DefeasibleConflictDepth,
)

lines = []
for level in (0, 2, 5):
    task = DefeasibleConflictDepth()
    task.config.set_level(level)
    lines.append(f"## Level {level}")
    picked = []
    while len(picked) < 2:
        ex = task.generate_example()
        if picked and ex.answer == picked[0].answer:
            continue
        picked.append(ex)
    for ex in picked:
        prompt = task.render_prompt(ex.metadata)
        lines.append("### Prompt")
        lines.append(prompt)
        lines.append("### Answer")
        lines.append(ex.answer)
        lines.append("")

with open(os.path.join(BASE, "samples_M16.md"), "w") as f:
    f.write("\n".join(lines))
print("wrote samples_M16.md")
