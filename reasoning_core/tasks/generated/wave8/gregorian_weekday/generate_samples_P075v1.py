import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.gregorian_weekday.gregorian_weekday import (
    GregorianWeekday,
)

SEED = 4114080681


def main():
    random.seed(SEED)
    task = GregorianWeekday()
    out = Path(__file__).with_name("samples_P075v1.md")
    lines = ["# gregorian_weekday samples (P075v1)", ""]
    for level in (0, 2, 5):
        lines.append(f"### Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            entry = task.generate_example()
            lines.append("**Prompt:**")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append(entry.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
