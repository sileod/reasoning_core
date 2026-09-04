import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.union_find_trace.union_find_trace import UnionFindTrace


def main():
    random.seed(1602037825)
    task = UnionFindTrace()
    out = Path(__file__).with_name("samples_P019v1.md")
    sections = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        task.config.apply_difficulty(level)
        lines = ["## Level %d" % level, ""]
        for _ in range(2):
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            lines.append("### Prompt")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append(x.answer)
            lines.append("")
        sections.append("\n".join(lines))
        # reset config for next level
        task.config.set_level(level)
    out.write_text("\n".join(sections) + "\n")
    print(out)


main()
