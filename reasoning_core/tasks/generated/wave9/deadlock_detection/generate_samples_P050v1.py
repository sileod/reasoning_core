import random
from pathlib import Path

random.seed(4249108675)

from reasoning_core.tasks.generated.wave9.deadlock_detection.deadlock_detection import (
    DeadlockDetection,
)

HERE = Path(__file__).parent


def main():
    lines = []
    t = DeadlockDetection()
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        t.config.set_level(level)
        for idx in range(2):
            e = t.generate_example()
            lines.append(f"### Example {idx + 1}")
            lines.append("")
            lines.append(t.render_prompt(e.metadata))
            lines.append("")
            lines.append(f"**Answer:** {e.answer}")
            lines.append("")
    (HERE / "samples_P050v1.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
