import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.generic_variance_subtyping.generic_variance_subtyping import (
    VarianceSubtyping,
)

SEED = 2438869589


def main():
    random.seed(SEED)
    task = VarianceSubtyping()
    out = Path(__file__).with_name("samples_P056v1.md")
    lines = ["# Samples for P056v1 (generic_variance_subtyping)", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in (1, 2):
            ex = task.generate_example()
            lines.append(f"### Example {i}")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
