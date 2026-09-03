import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.struct_layout_alignment.struct_layout_alignment import (
    StructLayoutAlignment,
)

SEED = 527083921


def main():
    random.seed(SEED)
    task = StructLayoutAlignment()
    out = []
    for level in (0, 2, 5):
        out.append(f"## Level {level}")
        task.config.set_level(level)
        for i in range(2):
            ex = task.generate_example()
            out.append(f"### Example {i+1}")
            out.append("**Prompt:** " + task.render_prompt(ex.metadata))
            out.append("**Answer:** " + ex.answer)
            out.append("")
    path = Path(__file__).with_name("samples_P058v2.md")
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
