import random
from pathlib import Path

random.seed(2807339826)

from reasoning_core.tasks.generated.wave9.program_slice_dependencies.program_slice_dependencies import (
    ProgramSliceDependencies,
)

out = []
task = ProgramSliceDependencies()
for level, count in ((0, 2), (2, 2), (5, 2)):
    task.config.set_level(level)
    out.append(f"## Level {level}")
    for i in range(count):
        ex = task.generate_example()
        out.append(f"### Example {i + 1}")
        out.append(task.render_prompt(ex.metadata))
        out.append(f"**Answer:** {ex.answer}")
        out.append("")

out_path = Path(__file__).with_name("samples_P029v1.md")
with open(out_path, "w") as f:
    f.write("\n".join(out).rstrip() + "\n")

print("wrote samples_P029v1.md")
