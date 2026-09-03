import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.poset_cover_query.poset_cover_query import (
    PosetCoverQuery,
)

random.seed(2960624742)

OUT = Path(__file__).with_name("samples_P025v1.md")


def main():
    lines = []
    task = PosetCoverQuery()
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            lines.append(prompt.strip())
            lines.append("")
            lines.append(f"**Answer:** {x.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
