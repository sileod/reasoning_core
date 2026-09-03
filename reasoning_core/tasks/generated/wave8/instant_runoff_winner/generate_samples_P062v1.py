import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))

from reasoning_core.tasks.generated.wave8.instant_runoff_winner.instant_runoff_winner import (
    InstantRunoffWinner,
)

SEED = 1501836147
OUT = HERE / "samples_P062v1.md"


def main():
    random.seed(SEED)
    task = InstantRunoffWinner()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        for i in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append("Prompt:")
            lines.append(ex.prompt)
            lines.append("")
            lines.append(f"Answer: {ex.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
