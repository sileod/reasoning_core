"""Byte-reproducible sample generation for clock_page_replacement (P045v1)."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.clock_page_replacement.clock_page_replacement import (
    ClockPageReplacement,
)

SEED = 2635584578
OUT = Path(__file__).with_name("samples_P045v1.md")


def main():
    random.seed(SEED)
    task = ClockPageReplacement()
    lines = [
        "# Sample gallery: clock_page_replacement (P045v1)",
        "",
        "Given a Clock (second-chance) page-replacement state and a page fault, "
        "output the frame index chosen for eviction.",
        "",
    ]
    for level in (0, 2, 5):
        for _ in range(2):
            task.config.set_level(level)
            e = task.generate_example()
            lines.append(f"## Level {level}")
            lines.append("")
            lines.append("### Prompt")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
