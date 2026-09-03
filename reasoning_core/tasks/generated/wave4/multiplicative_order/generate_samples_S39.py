import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s39_multiplicative_order.s39_multiplicative_order import (
    MultiplicativeOrder,
)

SEED = 131861191
OUT = Path(__file__).with_name("samples_S39.md")


def main():
    random.seed(SEED)
    task = MultiplicativeOrder()
    lines = ["# samples_S39", "", "## Level 0", ""]
    lines.append("### Example 1")
    lines.append("")
    e = task.generate_example()
    lines.append(f"**Prompt:** {task.render_prompt(e.metadata)}")
    lines.append("")
    lines.append(f"**Answer:** {e.answer}")
    lines.append("")
    lines.append("### Example 2")
    lines.append("")
    e = task.generate_example()
    lines.append(f"**Prompt:** {task.render_prompt(e.metadata)}")
    lines.append("")
    lines.append(f"**Answer:** {e.answer}")
    lines.append("")
    lines.append("## Level 2")
    lines.append("")
    task.config.set_level(2)
    for i in (1, 2):
        lines.append(f"### Example {i}")
        lines.append("")
        e = task.generate_example()
        lines.append(f"**Prompt:** {task.render_prompt(e.metadata)}")
        lines.append("")
        lines.append(f"**Answer:** {e.answer}")
        lines.append("")
    lines.append("## Level 5")
    lines.append("")
    task.config.set_level(5)
    for i in (1, 2):
        lines.append(f"### Example {i}")
        lines.append("")
        e = task.generate_example()
        lines.append(f"**Prompt:** {task.render_prompt(e.metadata)}")
        lines.append("")
        lines.append(f"**Answer:** {e.answer}")
        lines.append("")
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
