import random
from pathlib import Path

random.seed(236351923)

from reasoning_core.tasks.generated.wave8.two_phase_lock_blocker.two_phase_lock_blocker import (
    TwoPhaseLockBlocker,
)

OUT = Path(__file__).with_name("samples_P034v1.md")


def main():
    task = TwoPhaseLockBlocker()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append("## Level %d" % level)
        lines.append("")
        for _ in range(2):
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            lines.append(prompt)
            lines.append("")
            lines.append("Answer: %s" % entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
