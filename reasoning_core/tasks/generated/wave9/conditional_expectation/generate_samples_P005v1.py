import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.conditional_expectation.conditional_expectation import (
    ConditionalExpectation,
)

SEED = 729651269
OUT = Path(__file__).with_name("samples_P005v1.md")


def main():
    random.seed(SEED)
    task = ConditionalExpectation()
    lines = ["# Conditional expectation samples (P005v1)", ""]
    for level in (0, 2, 5):
        lines.append("## Level %d" % level)
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            lines.append("### Prompt")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append(entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
