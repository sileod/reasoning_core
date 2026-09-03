import random
import sys
from pathlib import Path

sys.setrecursionlimit(10000)
random.seed(1493643473)

from reasoning_core.tasks.generated.wave8.c3_linearization.c3_linearization import (
    C3Linearization,
)

OUT = Path(__file__).with_name("samples_P054v2.md")

task = C3Linearization()
lines = ["# C3 Linearization samples (P054v2)", ""]
for level, n in ((0, 2), (2, 2), (5, 2)):
    lines.append(f"## Level {level}")
    lines.append("")
    for _ in range(n):
        ex = task.generate_example(level=level)
        lines.append("Prompt:")
        lines.append(ex.prompt)
        lines.append("")
        lines.append("Answer:")
        lines.append(ex.answer)
        lines.append("")
        lines.append("---")
        lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
