import random
from pathlib import Path

random.seed(1245131062)

from reasoning_core.tasks.generated.wave8.crc_remainder.crc_remainder import CrcRemainder

OUT = Path(__file__).with_name("samples_P068v1.md")


def emit(task, level):
    task.config.set_level(level)
    task.config.apply_difficulty(level)
    return task.generate_example()


def main():
    task = CrcRemainder()
    lines = []
    for level in (0, 2, 5):
        lines.append("## Level {}".format(level))
        for _ in range(2):
            ex = emit(task, level)
            lines.append("### Example")
            lines.append("**Prompt:**")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("**Answer:**")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
