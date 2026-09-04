import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.data_lineage_trace.data_lineage_trace import DataLineageTrace


def main():
    random.seed(895421081)
    task = DataLineageTrace()
    out = Path(__file__).with_name("samples_P037v1.md")

    lines = []
    lines.append("# Data Lineage Trace samples (P037v1)")
    lines.append("")
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(2):
            ex = task.generate_example(level=level)
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append(ex.prompt)
            lines.append("")
            lines.append("**Answer**: " + ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
