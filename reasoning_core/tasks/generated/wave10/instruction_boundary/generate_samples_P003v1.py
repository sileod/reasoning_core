"""Generate samples_P003v1.md for the instruction_data_boundary trial."""
import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.instruction_data_boundary import (
    instruction_data_boundary as mod,
)

SEED = 2129890259


def main():
    random.seed(SEED)
    task = mod.InstructionBoundary()
    out = Path(__file__).with_name("samples_P003v1.md")
    lines = ["# Samples: instruction_data_boundary", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        cfg = mod.InstructionBoundaryConfig()
        cfg.set_level(level)
        task.config = cfg
        for _ in range(2):
            e = task.generate_example()
            prompt = task.render_prompt(e.metadata)
            lines.append(prompt)
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
