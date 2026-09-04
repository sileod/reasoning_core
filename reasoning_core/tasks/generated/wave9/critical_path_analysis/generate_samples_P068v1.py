import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.critical_path_analysis.critical_path_analysis import CriticalPathAnalysis

SEED = 1776440515


def main():
    random.seed(SEED)
    out = Path(__file__).with_name("samples_P068v1.md")
    task = CriticalPathAnalysis()
    lines = ["# Samples for critical_path_analysis (P068v1)", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        for k in range(2):
            x = task.generate_example(level=level)
            lines.append(f"### Example {k + 1}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            for pl in task.render_prompt(x.metadata).splitlines():
                lines.append(f"    {pl}")
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(f"    {x.answer}")
            lines.append("")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
