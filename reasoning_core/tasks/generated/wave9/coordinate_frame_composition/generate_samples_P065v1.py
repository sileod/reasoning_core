import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.coordinate_frame_composition.coordinate_frame_composition import (
    CoordinateFrameComposition,
)


def main():
    random.seed(2972124136)
    out = Path(__file__).with_name("samples_P065v1.md")
    lines = []
    for level in (0, 2, 5):
        task = CoordinateFrameComposition(_level=level)
        lines.append("## Level {}".format(level))
        lines.append("")
        for _ in range(2):
            ex = task.generate_example(level=level)
            lines.append("### Example")
            lines.append("")
            lines.append("**Prompt**:")
            lines.append("")
            lines.append(ex.prompt)
            lines.append("")
            lines.append("**Answer**:")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
