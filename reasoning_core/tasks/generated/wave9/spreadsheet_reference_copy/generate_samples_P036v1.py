import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.spreadsheet_reference_copy import spreadsheet_reference_copy as m

SEED = 3518700511


def main():
    random.seed(SEED)
    task = m.SpreadsheetReferenceCopy()
    out = []
    out.append("# Samples for spreadsheet_reference_copy (P036v1)\n")
    for level in (0, 2, 5):
        out.append(f"## Level {level}\n")
        for _ in range(2):
            ex = task.generate_example(level=level)
            out.append(task.render_prompt(ex.metadata))
            out.append("")
            out.append("Answer:")
            out.append(ex.answer)
            out.append("")
    path = Path(__file__).with_name("samples_P036v1.md")
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
