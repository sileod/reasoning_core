"""Generate samples_P015v2.md with reproducible seeded examples."""
import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.presupposition_repair.presupposition_repair import (
    PresuppositionRepair,
    PresuppositionRepairV2Config,
)

SEED = 3360584995
OUT = Path(__file__).with_name("samples_P015v2.md")


def main():
    random.seed(SEED)
    task = PresuppositionRepair()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config = PresuppositionRepairV2Config()
        task.config.set_level(level)
        for _ in range(2):
            entry = task.generate_example()
            lines.append("### Prompt")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("### Answer")
            lines.append(entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
