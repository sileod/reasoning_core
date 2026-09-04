import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.hierarchical_rollup.hierarchical_rollup import HierarchicalRollup


def main():
    random.seed(502759112)
    task = HierarchicalRollup()
    out_path = Path(__file__).with_name("samples_P039v1.md")
    lines = ["# HierarchicalRollup samples", ""]
    for level in (0, 2, 5):
        lines.append("## Level {}".format(level))
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_entry()
            prompt = task.render_prompt(e.metadata)
            lines.append("**Example {}**".format(i + 1))
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("Answer: {}".format(e.answer))
            lines.append("")
    out_path.write_text("\n".join(lines))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
