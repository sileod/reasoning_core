import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.heap_key_update.heap_key_update import (
    HeapKeyUpdate
)

SEED = 2479838412

LEVELS = {
    0: 2,
    2: 2,
    5: 2,
}


def main():
    random.seed(SEED)
    task = HeapKeyUpdate()
    out = Path(__file__).with_name("samples_P009v1.md")
    lines = ["# Samples for heap_key_update (P009v1)", ""]
    for level, count in LEVELS.items():
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(count):
            ex = task.generate_example()
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
