import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.hidden_state_filtering.hidden_state_filtering import (
    HiddenStateFiltering,
)


def main():
    random.seed(1475571465)
    task = HiddenStateFiltering()
    out = Path(__file__).with_name("samples_P002v1.md")
    lines = ["# hidden_state_filtering samples", ""]
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            ex = task.generate_example()
            lines.append(ex.prompt)
            lines.append("")
            lines.append("**Answer:** " + ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
