import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.lossless_binary_decomposition.lossless_binary_decomposition import (
    LosslessBinaryDecomposition,
)


def main():
    random.seed(2705364208)
    task = LosslessBinaryDecomposition()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        for i in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {i + 1}")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append(f"Answer: {ex.answer}")
            lines.append("")
    Path(__file__).with_name("samples_P030v1.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
