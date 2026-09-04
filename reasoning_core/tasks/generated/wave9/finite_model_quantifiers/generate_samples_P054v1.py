import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.finite_model_quantifiers.finite_model_quantifiers import (
    FiniteModelQuantifiers,
)

SEED = 2104874763
LEVELS = [0, 2, 5]
PER_LEVEL = 2


def main():
    random.seed(SEED)
    task = FiniteModelQuantifiers()
    out = ["# samples_P054v1", ""]
    for level in LEVELS:
        task.config.set_level(level)
        out.append(f"## Level {level}")
        out.append("")
        for i in range(PER_LEVEL):
            ex = task.generate_example()
            out.append(f"### Example {i+1}")
            out.append("")
            out.append("**Prompt:**")
            out.append("")
            out.append(task.render_prompt(ex.metadata))
            out.append("")
            out.append("**Answer:**")
            out.append("")
            out.append(ex.answer)
            out.append("")
    path = Path(__file__).with_name("samples_P054v1.md")
    path.write_text("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
