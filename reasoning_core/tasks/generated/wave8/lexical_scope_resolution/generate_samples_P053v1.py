import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.lexical_scope_resolution.lexical_scope_resolution import (
    LexicalScopeResolution,
)


def main():
    random.seed(3759181579)
    out_path = Path(__file__).with_name("samples_P053v1.md")
    task = LexicalScopeResolution()
    lines = []
    for level in [0, 2, 5]:
        lines.append("## Level %d" % level)
        task.config.set_level(level)
        for idx in range(2):
            entry = task.generate_example()
            lines.append("### Example %d" % (idx + 1))
            lines.append("**Prompt:**")
            lines.append(entry.metadata["prompt"] if "prompt" in entry.metadata else task.render_prompt(entry.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append(entry.answer)
            lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
