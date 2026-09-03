import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.quorum_intersection.quorum_intersection import (
    QuorumIntersection,
)

random.seed(4167519577)

OUT = Path(__file__).with_name("samples_P040v1.md")


def main():
    t = QuorumIntersection()
    lines = []
    for level in (0, 2, 5):
        t.config.set_level(level)
        lines.append(f"## Level {level}")
        for i in range(2):
            e = t.generate_example()
            prompt = t.render_prompt(e.metadata)
            lines.append(f"### Example {i + 1}")
            lines.append("Prompt:")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
