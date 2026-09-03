import random
from pathlib import Path
from reasoning_core.tasks.generated.wave8.union_find_representative.union_find_representative import (
    UnionFindRepresentative,
)

SEED = 3837604770
OUT = Path(__file__).with_name("samples_P014v1.md")


def main():
    random.seed(SEED)
    task = UnionFindRepresentative()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for _ in range(2):
            x = task.generate_example()
            lines.append("```")
            lines.append(task.render_prompt(x.metadata))
            lines.append("```")
            lines.append("")
            lines.append(f"Answer: {x.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
