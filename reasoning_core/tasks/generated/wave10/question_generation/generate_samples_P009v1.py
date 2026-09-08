import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.question_generation.question_generation import (
    QuestionGeneration,
)

SEED = 3704315384
OUT = Path(__file__).with_name("samples_P009v1.md")


def main():
    random.seed(SEED)
    task = QuestionGeneration()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}\n")
        for i in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {i+1}\n")
            lines.append(f"**Prompt:**\n\n```\n{task.render_prompt(ex.metadata)}\n```\n")
            lines.append(f"**Answer:**\n\n```\n{ex.answer}\n```\n")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
