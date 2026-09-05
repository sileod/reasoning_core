import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.unix_mode_permission.unix_mode_permission import (
    UnixModePermission,
)

SEED = 812548150
LEVELS = (0, 2, 5)
EXAMPLES_PER_LEVEL = 2


def main():
    random.seed(SEED)
    task = UnixModePermission()
    out = Path(__file__).with_name("samples_P047v1.md")
    lines = ["# UnixModePermission - samples P047v1", ""]
    for level in LEVELS:
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(EXAMPLES_PER_LEVEL):
            ex = task.generate_example()
            lines.append(ex.prompt)
            lines.append("")
            lines.append(f"Answer: {ex.answer}")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
