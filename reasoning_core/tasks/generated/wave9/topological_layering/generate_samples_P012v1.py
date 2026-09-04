import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.topological_layering.topological_layering import (
    TopologicalLayering,
)


def main():
    random.seed(1277236794)
    task = TopologicalLayering()
    out = Path(__file__).with_name("samples_P012v1.md")
    lines = ["# Samples for P012v1\n"]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}\n")
        for _ in range(2):
            task.config.set_level(level)
            e = task.generate_example()
            lines.append("### Prompt")
            lines.append("```")
            lines.append(task.render_prompt(e.metadata))
            lines.append("```")
            lines.append("### Answer")
            lines.append("```")
            lines.append(e.answer)
            lines.append("```")
            lines.append("")
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
