import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.missing_dimension.missing_dimension import (
    MissingDimension,
)

SEED = 2952937494
OUT = Path(__file__).with_name("samples_P074v1.md")


def main():
    random.seed(SEED)
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}\n")
        task = MissingDimension()
        task.config.set_level(level)
        for k in range(3):
            e = task.generate_example()
            lines.append(e.prompt)
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
