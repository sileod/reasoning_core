import random
from pathlib import Path

from reasoning_core.tasks.generated.wave3.s26_continued_fractions.task import (
    ContinuedFractions,
    ContinuedFractionsConfig,
)

SEED = 3369804028

OUT = Path(__file__).with_name("samples_S26.md")

LEVELS = [0, 2, 5]
PER_LEVEL = 2


def main():
    random.seed(SEED)
    lines = []
    for level in LEVELS:
        lines.append("## Level %d" % level)
        cfg = ContinuedFractionsConfig()
        cfg.set_level(level)
        task = ContinuedFractions(config_cls=type(cfg), config=cfg)
        for _ in range(PER_LEVEL):
            entry = task.generate_example()
            lines.append("### Prompt")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("### Answer")
            lines.append(entry.answer)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
