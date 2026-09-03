"""Generate samples_P039v1.md for wait_for_deadlock."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.wait_for_deadlock.wait_for_deadlock import (
    WaitForDeadlock,
)

SEED = 1977587174

OUT = Path(__file__).with_name("samples_P039v1.md")

LEVELS = [(0, 2), (2, 2), (5, 2)]


def main():
    random.seed(SEED)
    task = WaitForDeadlock()
    lines = [f"# Samples wait_for_deadlock (seed {SEED})\n"]
    for level, count in LEVELS:
        lines.append(f"\n## Level {level}\n")
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        for _ in range(count):
            entry = task.generate_entry()
            prompt = task.render_prompt(entry.metadata)
            lines.append("### Example\n")
            lines.append("Prompt:\n")
            lines.append("```\n" + prompt + "\n```\n")
            lines.append("Answer:\n")
            lines.append("```\n" + entry.answer + "\n```\n")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
