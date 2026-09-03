import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.nominal_subtyping.nominal_subtyping import (
    NominalSubtyping, NominalSubtypingConfig,
)

SEED = 823763445
OUT = Path(__file__).with_name("samples_P055v2.md")


def main():
    random.seed(SEED)
    t = NominalSubtyping(config=NominalSubtypingConfig())
    lines = ["# Nominal Subtyping v2 samples", ""]
    for level in (0, 2, 5):
        lines.append("## Level %d" % level)
        lines.append("")
        t.config.set_level(level)
        for i in range(2):
            ex = t.generate_example()
            lines.append("### Example %d" % (i + 1))
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(t.render_prompt(ex.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
