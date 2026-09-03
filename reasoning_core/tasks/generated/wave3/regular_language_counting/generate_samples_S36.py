import random
from pathlib import Path

from reasoning_core.tasks.generated.wave3.s36_regular_language_counting.s36_regular_language_counting import (
    RegularLanguageCounting,
)


def main():
    random.seed(168153944)
    lines = []
    task = RegularLanguageCounting()
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for i in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append(f"**Prompt:** {ex.metadata['_prompt'] if '_prompt' in ex.metadata else task.render_prompt(ex.metadata)}")
            lines.append(f"**Answer:** {ex.answer}")
            lines.append("")

    script_dir = Path(__file__).with_name("samples_S36.md")
    script_dir.write_text("\n".join(lines) + "\n")
    print("wrote samples_S36.md")


if __name__ == "__main__":
    main()
