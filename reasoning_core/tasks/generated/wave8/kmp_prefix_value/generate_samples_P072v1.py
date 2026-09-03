import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.kmp_prefix_value.kmp_prefix_value import KmpPrefixValue

SEED = 161918849
OUT = Path(__file__).with_name("samples_P072v1.md")


def main():
    random.seed(SEED)
    task = KmpPrefixValue()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"# Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            entry = task.generate_entry()
            prompt = task.render_prompt(entry.metadata)
            lines.append("**Prompt:**")
            lines.append(prompt)
            lines.append("")
            lines.append("**Answer:**")
            lines.append(entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
