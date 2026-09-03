import random
import sys
from pathlib import Path

from reasoning_core.tasks.generated.wave8.finite_group_inverse.finite_group_inverse import (
    FiniteGroupInverse,
)

SEED = 4262150195
OUT = Path(__file__).with_name("samples_P019v1.md")
LEVELS = (0, 2, 5)
PER_LEVEL = 2

random.seed(SEED)
task = FiniteGroupInverse()


def main():
    lines = ["# finite_group_inverse samples", ""]
    for level in LEVELS:
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(PER_LEVEL):
            task.config.set_level(level)
            e = task.generate_example()
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
