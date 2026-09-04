import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.three_dimensional_orientation.three_dimensional_orientation import (
    ThreeDimensionalOrientation,
    ThreeDimensionalOrientationConfig,
)

random.seed(2939315147)
OUT = Path(__file__).with_name("samples_P066v1.md")


def write():
    out = []
    out.append("# Samples P066v1 - three_dimensional_orientation")
    out.append("")
    for level in (0, 2, 5):
        out.append(f"## Level {level}")
        out.append("")
        cfg = ThreeDimensionalOrientationConfig()
        cfg.set_level(level)
        t = ThreeDimensionalOrientation(config=cfg)
        for _ in range(2):
            ex = t.generate_example()
            prompt = t.render_prompt(ex.metadata)
            out.append("### Example")
            out.append("")
            out.append("**Prompt:**")
            out.append("")
            out.append(prompt)
            out.append("")
            out.append("**Answer:**")
            out.append("")
            out.append(ex.answer)
            out.append("")
    OUT.write_text("\n".join(out))
    print(OUT)


if __name__ == "__main__":
    write()
