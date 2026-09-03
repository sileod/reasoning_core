import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.dhondt_apportionment.dhondt_apportionment import (
    DhondtApportionment,
)

SEED = 3332846379
OUT = Path(__file__).with_name("samples_P064v1.md")


def main():
    random.seed(SEED)
    t = DhondtApportionment()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        t.config.set_level(level)
        for _ in range(2):
            e = t.generate_example()
            lines.append("### Prompt")
            lines.append("")
            lines.append(t.render_prompt(e.metadata))
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
