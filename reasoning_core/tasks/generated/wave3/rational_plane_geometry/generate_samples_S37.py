import random
from pathlib import Path

from reasoning_core.tasks.generated.wave3.s37_rational_plane_geometry.s37_rational_plane_geometry import (
    RationalPlaneGeometry,
)

random.seed(2090976468)

OUT = Path(__file__).with_name("samples_S37.md")


def build(level, n=2):
    t = RationalPlaneGeometry()
    t.config.set_level(level)
    lines = [f"## Level {level}"]
    for _ in range(n):
        e = t.generate_example()
        lines.append("```")
        lines.append(t.render_prompt(e.metadata))
        lines.append("```")
        lines.append(f"**Answer:** {e.answer}")
    return lines


def main():
    parts = ["# S37 samples\n"]
    for lvl in (0, 2, 5):
        parts.append("\n".join(build(lvl)))
    OUT.write_text("\n".join(parts) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
