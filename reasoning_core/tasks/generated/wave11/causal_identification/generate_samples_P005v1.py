import random
from pathlib import Path

from reasoning_core.tasks.generated.wave11.causal_identification.causal_identification import (
    CausalIdentification,
)


def main():
    random.seed(3090820539)
    task = CausalIdentification()
    out_path = Path(__file__).with_name("samples_P005v1.md")
    lines = [
        "# Samples for P005v1: causal_identification",
        "",
        "Seed: 3090820539",
        "",
    ]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            e = task.generate_example()
            lines.append("### Prompt")
            lines.append("")
            lines.append("```")
            lines.append(task.render_prompt(e.metadata))
            lines.append("```")
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append("```")
            lines.append(e.answer)
            lines.append("```")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
