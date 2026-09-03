"""Generate samples_P080v1.md for the semantic_version_precedence task."""
import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.semantic_version_precedence.semantic_version_precedence import (
    SemanticVersionPrecedence,
)

random.seed(157335281)

OUT = Path(__file__).with_name("samples_P080v1.md")


def emit_level(lines, level, n=2):
    task = SemanticVersionPrecedence()
    task.config.set_level(level)
    lines.append(f"## Level {level}")
    lines.append("")
    for _ in range(n):
        e = task.generate_example()
        lines.append("### Example")
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append(task.render_prompt(e.metadata))
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(e.answer)
        lines.append("")
    lines.append("")


def main():
    lines = ["# semantic_version_precedence samples", ""]
    for level in (0, 2, 5):
        emit_level(lines, level)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
