import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.exact_rounding_pipeline.exact_rounding_pipeline import (
    ExactRoundingPipeline, ExactRoundingPipelineConfig)

random.seed(3948247022)
OUT = Path(__file__).with_name("samples_P070v1.md")
TASK = ExactRoundingPipeline()
TASK.seed = 3948247022
TASK.config.seed = 3948247022

lines = []
for level in (0, 2, 5):
    TASK.config.set_level(level)
    lines.append(f"\n## Level {level}\n")
    for i in range(2):
        e = TASK.generate_example()
        lines.append(f"### Example {i + 1}\n")
        lines.append(TASK.render_prompt(e.metadata))
        lines.append("\nAnswer:")
        lines.append(e.answer)
        lines.append("")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
