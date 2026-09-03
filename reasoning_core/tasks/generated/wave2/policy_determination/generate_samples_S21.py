import random
from pathlib import Path

from reasoning_core.tasks.generated.wave2.s21_policy_determination.policy_determination import (
    PolicyDetermination,
    PolicyConfig,
)

SEED = 3335976966
OUT = Path(__file__).parent / "samples_S21.md"


def main():
    random.seed(SEED)
    task = PolicyDetermination()
    lines = []
    for level in (0, 2, 5):
        task.config = PolicyConfig()
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(2):
            entry = task.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append("")
            lines.append("**Prompt**")
            lines.append("")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("")
            lines.append("**Answer**")
            lines.append("")
            lines.append(entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
