import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.virtual_method_dispatch.virtual_method_dispatch import (
    VirtualMethodDispatch,
)

SEED = 480048984


def main():
    random.seed(SEED)
    task = VirtualMethodDispatch()
    lines = ["# Virtual Method Dispatch - P059v1 samples"]
    for level in (0, 2, 5):
        lines.append("")
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(2):
            task.config.set_level(level)
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append("```")
            lines.append(entry.answer)
            lines.append("```")
            lines.append("")
    out = Path(__file__).with_name("samples_P059v1.md")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
