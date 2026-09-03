from pathlib import Path

import random

from reasoning_core.tasks.generated.wave8.group_homomorphism_check.group_homomorphism_check import (
    GroupHomomorphismCheck,
)

SEED = 1330648825


def main():
    random.seed(SEED)
    task = GroupHomomorphismCheck()
    out = Path(__file__).with_name("samples_P023v1.md")
    lines = ["# Samples for group_homomorphism_check", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            ex = task.generate_example()
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("Answer:")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
