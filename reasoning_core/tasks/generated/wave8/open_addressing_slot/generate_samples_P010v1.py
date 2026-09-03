import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.open_addressing_slot.open_addressing_slot import (
    OpenAddressingSlot,
)

SEED = 64225588
LEVELS = [0, 2, 5]
PER_LEVEL = 2


def main():
    random.seed(SEED)
    task = OpenAddressingSlot()
    lines = []
    for level in LEVELS:
        lines.append("")
        lines.append("## Level %d" % level)
        task.config.set_level(level)
        for _ in range(PER_LEVEL):
            ex = task.generate_example()
            lines.append("")
            lines.append("### Example")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("Answer: %s" % ex.answer)
    out = Path(__file__).with_name("samples_P010v1.md")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
