from pathlib import Path

import random

from reasoning_core.tasks.generated.wave9.query_plan_execution.query_plan_execution import QueryPlanExecution


def main():
    random.seed(390250470)
    out = Path(__file__).with_name("samples_P040v1.md")
    task = QueryPlanExecution()
    lines = ["# query_plan_execution samples (P040v1)", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(2):
            ex = task.generate_example(level=level)
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
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
