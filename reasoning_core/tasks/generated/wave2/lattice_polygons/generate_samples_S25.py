import os
import random

from reasoning_core.tasks.generated.wave2.s25_lattice_polygons.s25_lattice_polygons import (
    LatticePolygons,
)

SEED = 3944590077

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    random.seed(SEED)
    task = LatticePolygons()
    lines = []
    for level in [0, 2, 5]:
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            ex = task.generate_example(level=level)
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(ex.prompt)
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(str(ex.answer))
            lines.append("")
    with open(os.path.join(HERE, "samples_S25.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
