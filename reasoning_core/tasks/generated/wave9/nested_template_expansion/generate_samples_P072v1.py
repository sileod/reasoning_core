import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.nested_template_expansion.nested_template_expansion import (
    NestedTemplateExpansion,
)


def main():
    random.seed(2045813452)
    out = Path(__file__).with_name("samples_P072v1.md")
    lines = []
    task = NestedTemplateExpansion()
    for level in (0, 2, 5):
        lines.append("# Level %d" % level)
        lines.append("")
        task.config.set_level(level)
        for idx in range(2):
            ex = task.generate_example()
            lines.append("## Example %d" % (idx + 1))
            lines.append("")
            lines.append("### Prompt")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("### Answer")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
