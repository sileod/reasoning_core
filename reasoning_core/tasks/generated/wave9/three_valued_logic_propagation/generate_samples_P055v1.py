import random
from pathlib import Path

random.seed(1851018107)

from reasoning_core.tasks.generated.wave9.three_valued_logic_propagation. \
    three_valued_logic_propagation import ThreeValuedLogicPropagation

OUT = Path(__file__).with_name("samples_P055v1.md")


def main():
    task = ThreeValuedLogicPropagation()
    lines = ["# samples_P055v1", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        for ex in range(2):
            task.config.set_level(level)
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            lines.append(f"### Example {ex + 1}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(entry.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
