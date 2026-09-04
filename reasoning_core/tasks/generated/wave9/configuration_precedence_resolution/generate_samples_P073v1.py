import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.configuration_precedence_resolution.configuration_precedence_resolution import (
    ConfigurationPrecedenceResolution,
)

SEED = 1903761430
OUT = Path(__file__).with_name("samples_P073v1.md")


def main():
    random.seed(SEED)
    task = ConfigurationPrecedenceResolution()
    lines = ["# ConfigurationPrecedenceResolution samples (P073v1)\n"]
    for level in (0, 2, 5):
        lines.append("## Level %d\n" % level)
        for i in range(2):
            task.config.set_level(level)
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            lines.append("### Example %d\n" % (i + 1))
            lines.append("**Prompt:**\n")
            lines.append(prompt + "\n")
            lines.append("**Answer:**\n\n" + entry.answer + "\n")
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
