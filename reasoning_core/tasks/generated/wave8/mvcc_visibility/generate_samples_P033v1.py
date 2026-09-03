import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.mvcc_visibility.mvcc_visibility import (
    MVCCVisibility,
)

SEED = 1305167045


def main():
    random.seed(SEED)
    task = MVCCVisibility()
    out = Path(__file__).with_name("samples_P033v1.md")
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append("**Prompt:**")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append(e.answer)
            lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
