import random
from pathlib import Path

from equation_balancing import EquationBalancing

SEED = 16602065
OUT = Path(__file__).with_name("samples_S45.md")


def main():
    random.seed(SEED)
    task = EquationBalancing()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            lines.append(prompt)
            lines.append("")
            lines.append(f"Answer: {entry.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
