import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.causal_intervention.causal_intervention import (
    CausalIntervention,
)


def main():
    random.seed(3320301215)
    task = CausalIntervention()
    out = Path(__file__).with_name("samples_P008v1.md")
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            entry = task.generate_example()
            lines.append("### Example %d" % (i + 1))
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("```")
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append("```")
            lines.append(entry.answer)
            lines.append("```")
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
