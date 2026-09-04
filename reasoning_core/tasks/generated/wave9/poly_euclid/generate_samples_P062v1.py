"""Generate the reproducible sample file samples_P062v1.md for the trial."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.polynomial_euclidean_algorithm.poly_euclid_algorithm import (
    PolyEuclid,
)

SEED = 4112812501
OUT = Path(__file__).with_name("samples_P062v1.md")


def main():
    random.seed(SEED)
    lines = []
    for lvl in (0, 2, 5):
        lines.append(f"# Level {lvl}")
        t = PolyEuclid()
        t.config.seed = SEED
        t.config.set_level(lvl)
        for i in range(2):
            e = t.generate_example()
            lines.append(f"## Example {i + 1}")
            lines.append(f"Prompt:")
            lines.append(e.prompt)
            lines.append("")
            lines.append(f"Answer:")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
