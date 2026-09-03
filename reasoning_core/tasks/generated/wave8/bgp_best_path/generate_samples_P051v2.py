import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.bgp_best_path.bgp_best_path import BgpBestPath


def main():
    random.seed(3767003075)
    task = BgpBestPath()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        for i in range(2):
            entry = task.generate_example()
            lines.append("Level %d" % level)
            lines.append("Example %d" % (i + 1))
            lines.append("Prompt:")
            lines.append(entry.metadata.get("_prompt", task.render_prompt(entry.metadata)))
            lines.append("Answer: %s" % entry.answer)
            lines.append("")
    out = Path(__file__).with_name("samples_P051v2.md")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
