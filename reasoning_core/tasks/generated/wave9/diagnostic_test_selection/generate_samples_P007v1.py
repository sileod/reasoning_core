"""Generate samples_P007v1.md for the DiagnosticTestSelection task."""
import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.diagnostic_test_selection.diagnostic_test_selection import (
    DiagnosticTestSelection,
)

random.seed(1749059983)

OUT = Path(__file__).with_name("samples_P007v1.md")


def main():
    task = DiagnosticTestSelection()
    lines = ["# DiagnosticTestSelection samples (P007v1)", ""]
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            x = task.generate_example()
            lines.append("### Prompt")
            lines.append("")
            for line in task.render_prompt(x.metadata).splitlines():
                lines.append(f"    {line}")
            lines.append("")
            lines.append("**Answer:** " + x.answer)
            lines.append("")
            lines.append("---")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
