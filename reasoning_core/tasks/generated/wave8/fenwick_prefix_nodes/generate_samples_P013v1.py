import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.fenwick_prefix_nodes.fenwick_prefix_nodes import (
    FenwickPrefixNodes,
)


def main():
    random.seed(2872449786)
    out = Path(__file__).with_name("samples_P013v1.md")
    lines = []
    for level in (0, 2, 5):
        task = FenwickPrefixNodes()
        task.config.set_level(level)
        lines.append(f"## Level {level}\n")
        for k in range(2):
            e = task.generate_example()
            prompt = task.render_prompt(e.metadata)
            lines.append(f"### Example {k + 1}\n")
            lines.append(f"**Prompt:**\n\n```\n{prompt}\n```\n")
            lines.append(f"**Answer:**\n\n```\n{e.answer}\n```\n")
        lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
