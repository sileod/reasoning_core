import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.strongly_connected_component.strongly_connected_component import (
    StronglyConnectedComponent,
)

SEED = 3057161243
OUT = Path(__file__).with_name("samples_P001v1.md")


def main():
    random.seed(SEED)
    lines = []
    for level in (0, 2, 5):
        task = StronglyConnectedComponent()
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            ex = task.generate_example(level=level)
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(ex.prompt)
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
