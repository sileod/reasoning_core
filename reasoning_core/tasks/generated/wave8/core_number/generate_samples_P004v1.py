import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.core_number.core_number import CoreNumber

SEED = 2202313084
LEVELS = [0, 2, 5]
PER_LEVEL = 2


def main():
    random.seed(SEED)
    task = CoreNumber()
    lines = []
    for level in LEVELS:
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for _ in range(PER_LEVEL):
            e = task.generate_example()
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    out = Path(__file__).with_name("samples_P004v1.md")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
