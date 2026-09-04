import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.modular_constraint_solver.modular_constraint_solver import (
    ModularConstraint,
)


def main():
    random.seed(3591494931)
    task = ModularConstraint()
    out = Path(__file__).with_name("samples_P061v1.md")
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for _ in range(2):
            e = task.generate_example()
            lines.append("### Example")
            lines.append("**Prompt:**")
            lines.append("```")
            lines.append(task.render_prompt(e.metadata))
            lines.append("```")
            lines.append("**Answer:**")
            lines.append("```")
            lines.append(e.answer)
            lines.append("```")
            lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
