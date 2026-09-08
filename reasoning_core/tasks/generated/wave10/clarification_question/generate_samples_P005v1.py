"""Generate the samples markdown for clarification_question (trial P005v1)."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.clarification_question.clarification_question import (
    ClarificationQuestion,
)

SEED = 1817164334
OUT = Path(__file__).with_name("samples_P005v1.md")


def main():
    random.seed(SEED)
    task = ClarificationQuestion()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for _ in range(2):
            ex = task.generate_example()
            lines.append("Prompt:")
            lines.append("```")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("```")
            lines.append("")
            lines.append("Answer:")
            lines.append("```")
            lines.append(ex.answer)
            lines.append("```")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
