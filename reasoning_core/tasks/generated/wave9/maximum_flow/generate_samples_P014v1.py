import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.maximum_flow.maximum_flow import MaximumFlow

random.seed(3344734409)

OUT = Path(__file__).with_name("samples_P014v1.md")


def emit():
    task = MaximumFlow()
    lines = []
    for level, tag in [(0, "Level 0"), (2, "Level 2"), (5, "Level 5")]:
        lines.append(f"## {tag}")
        task.config.set_level(level)
        for i in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {i + 1}")
            lines.append("Prompt:")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    return "\n".join(lines)


def main():
    OUT.write_text(emit())


if __name__ == "__main__":
    main()
