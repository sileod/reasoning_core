import os
import random

from reasoning_core.tasks.generated.wave3.s34_linear_recurrences.s34_linear_recurrences import (
    LinearRecurrence,
    LinearRecurrenceConfig,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "samples_S34.md")
SEED = 1172182934


def main():
    random.seed(SEED)
    lines = ["# Samples for S34 linear_recurrences", ""]
    for level in (0, 2, 5):
        cfg = LinearRecurrenceConfig()
        cfg.set_level(level)
        task = LinearRecurrence(config=cfg)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            lines.append("### Prompt")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append(x.answer)
            lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
