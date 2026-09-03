import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.finite_group_element_order.finite_group_element_order import (
    FiniteGroupElementOrder,
)

SEED = 990936709


def main():
    random.seed(SEED)
    task = FiniteGroupElementOrder()
    lines = []
    levels = [0, 2, 5]
    for level in levels:
        lines.append(f"# Level {level}")
        lines.append("")
        task.config = type(task.config)()
        task.config.set_level(level)
        for i in range(2):
            entry = task.generate_example()
            lines.append(f"**Example {i + 1}**")
            lines.append("")
            lines.append("Prompt:")
            lines.append("")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append(entry.answer)
            lines.append("")
    out = Path(__file__).with_name("samples_P020v1.md")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
