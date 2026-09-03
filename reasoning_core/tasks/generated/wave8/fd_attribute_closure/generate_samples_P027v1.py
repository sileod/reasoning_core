import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.fd_attribute_closure.fd_attribute_closure import (
    AttributeClosure,
)


def main():
    random.seed(2514553751)
    out = Path(__file__).with_name("samples_P027v1.md")

    lines = []
    lines.append("# Samples P027v1 (fd_attribute_closure)")
    lines.append("")

    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task = AttributeClosure()
        task.config.set_level(level)
        for _ in range(3):
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

    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
