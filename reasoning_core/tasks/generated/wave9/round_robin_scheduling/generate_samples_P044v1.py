import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.round_robin_scheduling.round_robin_scheduling import (
    RoundRobinScheduling,
)

seed = 2693117635
random.seed(seed)

task = RoundRobinScheduling()
out = Path(__file__).with_name("samples_P044v1.md")
blocks = ["# round_robin_scheduling samples", f"seed {seed}", ""]

for level in (0, 2, 5):
    blocks.append(f"## Level {level}")
    for _ in range(2):
        task.config.set_level(level)
        e = task.generate_entry()
        blocks.append(
            "**Prompt:**\n```\n" + task.render_prompt(e.metadata) + "\n```"
        )
        blocks.append(
            f"**Answer:** `{e.answer}`\n"
        )
    blocks.append("")

out.write_text("\n".join(blocks))
print("wrote", out)
