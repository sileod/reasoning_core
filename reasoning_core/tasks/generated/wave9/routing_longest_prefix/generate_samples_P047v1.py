import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.routing_longest_prefix.routing_longest_prefix import (
    RoutingLongestPrefix,
)

SEED = 265960371
OUT = Path(__file__).with_name("samples_P047v1.md")


def main():
    random.seed(SEED)
    task = RoutingLongestPrefix()
    lines = []
    lines.append("# samples_P047v1 -- routing_longest_prefix")
    lines.append("")
    lines.append("Each entry shows the exact prompt the task emits and the gold next-hop answer underneath.")
    lines.append("")
    for level in (0, 2, 5):
        lines.append("## Level {}".format(level))
        lines.append("")
        for _ in range(2):
            task.config.seed = SEED
            task.config.set_level(level)
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            lines.append("**Prompt:**")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("")
            lines.append("**Answer:** {}".format(x.answer))
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
