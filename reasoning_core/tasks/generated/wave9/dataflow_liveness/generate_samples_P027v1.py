import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.dataflow_liveness.dataflow_liveness import (
    DataflowLiveness,
)


def _gen(task_cls, level, count, seed):
    random.seed(seed)
    c = task_cls.config_cls()
    c.set_level(level)
    task = task_cls()
    task.config = c
    return [task.generate_entry() for _ in range(count)]


def main():
    out = Path(__file__).with_name("samples_P027v1.md")
    task_cls = DataflowLiveness
    lines = []
    lines.append("# Samples P027v1\n")
    for level in (0, 2, 5):
        lines.append("## Level %d" % level)
        lines.append("")
        for i, e in enumerate(_gen(task_cls, level, 2, 1583885757 + level * 1000)):
            lines.append("### Example %d" % (i + 1))
            lines.append("")
            lines.append("**Prompt**")
            lines.append("")
            prompt = task_cls().render_prompt(e.metadata)
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("")
            lines.append("**Answer**")
            lines.append("")
            lines.append("```")
            lines.append(e.answer)
            lines.append("```")
            lines.append("")
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
