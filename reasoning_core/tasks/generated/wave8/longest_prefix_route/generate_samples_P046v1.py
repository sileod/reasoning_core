import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.longest_prefix_route.longest_prefix_route import (
    LongestPrefixRoute,
    LongestPrefixRouteConfig,
)

random.seed(1552420125)

OUT = Path(__file__).with_name("samples_P046v1.md")

task = LongestPrefixRoute()

lines = []
lines.append("# Samples for longest_prefix_route (P046v1)")
lines.append("")


def add_level(level):
    cfg = LongestPrefixRouteConfig()
    cfg.set_level(level)
    task.config = cfg
    lines.append(f"## Level {level}")
    lines.append("")
    for i in range(2):
        ex = task.generate_entry()
        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append("```")
        lines.append(task.render_prompt(ex.metadata))
        lines.append("```")
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append("```")
        lines.append(ex.answer)
        lines.append("```")
        lines.append("")


add_level(0)
add_level(2)
add_level(5)

OUT.write_text("\n".join(lines) + "\n")
print(OUT)
