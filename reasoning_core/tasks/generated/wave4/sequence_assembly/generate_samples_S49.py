import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s49_sequence_assembly.s49_sequence_assembly import (
    SequenceAssembly,
)


def main():
    random.seed(3175732428)
    out = Path(__file__).with_name("samples_S49.md")
    task = SequenceAssembly()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_example()
            lines.append(f"### Level {level} example {i + 1}")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append(f"**Answer:** {e.answer}")
            lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
