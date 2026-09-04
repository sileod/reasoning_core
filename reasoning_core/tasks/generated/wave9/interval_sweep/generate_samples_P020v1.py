import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from reasoning_core.tasks.generated.wave9.interval_sweep.interval_sweep import IntervalSweep

SEED = 2068888396
OUT = Path(__file__).with_name("samples_P020v1.md")

LEVELS = [0, 2, 5]
PER_LEVEL = 2


def main():
    random.seed(SEED)
    lines = []
    for level in LEVELS:
        lines.append(f"## Level {level}")
        t = IntervalSweep()
        t.config.set_level(level)
        for i in range(PER_LEVEL):
            e = t.generate_entry()
            lines.append(f"### Example {i + 1}")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(t.render_prompt(e.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
