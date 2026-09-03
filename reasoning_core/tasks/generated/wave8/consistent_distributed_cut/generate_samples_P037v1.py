import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.consistent_distributed_cut.consistent_distributed_cut import (
    ConsistentDistributedCut,
)

SEED = 3066933245
OUT = Path(__file__).with_name("samples_P037v1.md")

random.seed(SEED)


def render(level, n=2):
    task = ConsistentDistributedCut()
    task.config.set_level(level)
    lines = []
    for i in range(n):
        ex = task.generate_example()
        lines.append("")
        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append(task.render_prompt(ex.metadata))
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(ex.answer)
        lines.append("")
    return "\n".join(lines)


with OUT.open("w") as f:
    f.write("# samples_P037v1\n")
    for level in (0, 2, 5):
        f.write(f"\n## Level {level}\n")
        f.write(render(level))
    f.write("\n")
