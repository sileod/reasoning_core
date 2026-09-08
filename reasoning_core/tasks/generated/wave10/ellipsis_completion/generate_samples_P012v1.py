"""Generate the samples file for P012v1 (ellipsis_completion)."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.ellipsis_completion.ellipsis_completion import (
    EllipsisCompletion, EllipsisConfig,
)

SEED = 1777646155
OUT = Path(__file__).with_name("samples_P012v1.md")


def main():
    random.seed(SEED)
    lines = ["# Ellipsis Completion — P012v1 samples", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            t = EllipsisCompletion()
            t.config = EllipsisConfig()
            t.config.apply_difficulty(level)
            e = t.generate_entry()
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(t.render_prompt(e.metadata))
            lines.append("```")
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(f"`{e.answer}`")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
