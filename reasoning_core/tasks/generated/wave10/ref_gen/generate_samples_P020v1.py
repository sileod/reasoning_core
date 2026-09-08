import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.pragmatic_reference_generation.pragmatic_reference_generation import (
    RefGenTask,
)


def main():
    random.seed(3457176975)
    task = RefGenTask()
    out = []
    for level in (0, 2, 5):
        out.append(f"## Level {level}")
        out.append("")
        for _ in range(2):
            task.config.set_level(level)
            x = task.generate_example()
            out.append(task.render_prompt(x.metadata))
            out.append("")
            out.append(f"Answer: {x.answer}")
            out.append("")
    path = Path(__file__).with_name("samples_P020v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
