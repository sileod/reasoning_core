"""Byte-reproducible sample generator for trial P043v1."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.virtual_address_translation.virtual_address_translation import (
    VirtualAddressTranslation,
)

SEED = 28548294

OUT = Path(__file__).with_name("samples_P043v1.md")


def main():
    random.seed(SEED)
    task = VirtualAddressTranslation()
    lines = ["# Samples for P043v1: virtual_address_translation", ""]
    for level in [0, 2, 5]:
        task.config.set_level(level)
        lines.append("## Level {}".format(level))
        lines.append("")
        for i in range(2):
            ex = task.generate_example()
            lines.append("### Example {}".format(i + 1))
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
