import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.pivot_unpivot_transform.pivot_unpivot_transform import (
    PivotUnpivotTransform,
)

SEED = 3773979110

OUT = Path(__file__).with_name("samples_P034v1.md")


def build():
    random.seed(SEED)
    lines = []
    lines.append("# P034v1 samples: pivot_unpivot_transform")
    lines.append("")
    for level in [0, 2, 5]:
        task = PivotUnpivotTransform()
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(2):
            entry = task.generate_entry()
            prompt = task.render_prompt(entry.metadata)
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append("Prompt:")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append(entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    build()
