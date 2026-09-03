import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.topological_generation.topological_generation import (
    TopologicalGeneration,
)

OUT = Path(__file__).with_name("samples_P006v1.md")


def main():
    random.seed(3867398156)
    task = TopologicalGeneration()
    lines = ["# topological_generation samples (P006v1)", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            ex = task.generate_example()
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append(f"Answer: {ex.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
