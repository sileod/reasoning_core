import random
from pathlib import Path

random.seed(3867019559)

from reasoning_core.tasks.generated.wave9.counterfactual_twin_model.counterfactual_twin_model import (  # noqa: E402
    CounterfactualTwinModel,
)

OUT = Path(__file__).with_name("samples_P009v1.md")


def main():
    task = CounterfactualTwinModel()
    lines = []
    lines.append("# Counterfactual twin model samples (P009v1)")
    lines.append("")
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            ex = task.generate_example()
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append(f"Answer: {ex.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
