"""Generate samples_P002v1.md for the ArticulationVertices task."""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from articulation_vertices import ArticulationVertices

SEED = 3438477119
OUT = Path(__file__).with_name("samples_P002v1.md")


def main():
    random.seed(SEED)
    task = ArticulationVertices()
    lines = ["# ArticulationVertices samples (P002v1)", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_example()
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
