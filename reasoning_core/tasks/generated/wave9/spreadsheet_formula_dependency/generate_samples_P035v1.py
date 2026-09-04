import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.spreadsheet_formula_dependency.spreadsheet_formula_dependency import (
    SpreadsheetFormulaDependency,
)


def main():
    random.seed(780191338)
    out = Path(__file__).with_name("samples_P035v1.md")
    task = SpreadsheetFormulaDependency()
    lines = ["# Spreadsheet Formula Dependency - samples\n"]
    for level in (0, 2, 5):
        lines.append("## Level %d\n" % level)
        task.config.set_level(level)
        for _ in range(2):
            ex = task.generate_example()
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("```")
            lines.append("")
            lines.append("**Answer:** %s" % ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
