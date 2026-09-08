import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.minimal_edit.minimal_edit import MinimalEdit

SEED = 3529541070
OUT = Path(__file__).with_name("samples_P004v1.md")


def main():
    random.seed(SEED)
    task = MinimalEdit()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append("## Level {}".format(level))
        for i in range(2):
            e = task.generate_example()
            lines.append("### Example {}".format(i + 1))
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
