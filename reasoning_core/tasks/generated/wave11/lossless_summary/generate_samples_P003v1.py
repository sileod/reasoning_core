import random
from pathlib import Path

from reasoning_core.tasks.generated.wave11.lossless_summary.lossless_summary import (
    LosslessSummary,
)

SEED = 3797320393
OUT = Path(__file__).with_name("samples_P003v1.md")


def main():
    random.seed(SEED)
    lines = []
    task = LosslessSummary()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(2):
            example = task.generate_example()
            lines.append(f"## Level {level}")
            lines.append("")
            lines.append(task.render_prompt(example.metadata))
            lines.append("")
            lines.append(f"Answer: {example.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
