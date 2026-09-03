import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.vector_clock_order.vector_clock_order import (
    VectorClockOrder,
)


def main():
    random.seed(2985613391)
    out = Path(__file__).with_name("samples_P036v1.md")
    task = VectorClockOrder()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}\n")
        for _ in range(2):
            x = task.generate_example()
            lines.append("Prompt:\n")
            lines.append("```\n" + task.render_prompt(x.metadata) + "\n```\n")
            lines.append("Answer:\n")
            lines.append("```\n" + x.answer + "\n```\n")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
