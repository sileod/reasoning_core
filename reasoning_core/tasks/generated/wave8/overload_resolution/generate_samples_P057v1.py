import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.overload_resolution.overload_resolution import (
    OverloadResolution,
)

random.seed(3892076091)

OUT = Path(__file__).with_name("samples_P057v1.md")
t = OverloadResolution()

with open(OUT, "w") as f:
    f.write("# Samples P057v1: overload_resolution\n\n")
    for level in (0, 2, 5):
        f.write(f"## Level {level}\n\n")
        t.config.set_level(level)
        for _ in range(2):
            x = t.generate_example()
            f.write("Prompt:\n\n")
            f.write("```\n")
            f.write(t.render_prompt(x.metadata))
            f.write("\n```\n\n")
            f.write("Answer:\n\n")
            f.write(f"```\n{x.answer}\n```\n\n")
