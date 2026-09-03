import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.cron_next_fire.cron_next_fire import (
    CronNextFire,
)

SEED = 901858890
OUT = Path(__file__).with_name("samples_P076v1.md")


def main():
    random.seed(SEED)
    task = CronNextFire()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            x = task.generate_example()
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append("Prompt:")
            lines.append("")
            lines.append(x.prompt)
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append(x.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
