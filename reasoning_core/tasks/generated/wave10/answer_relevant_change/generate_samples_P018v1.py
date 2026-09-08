import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.answer_relevant_change.answer_relevant_change import (
    AnswerRelevantChange,
)


def main():
    random.seed(3783103188)
    task = AnswerRelevantChange()
    out = Path(__file__).with_name("samples_P018v1.md")
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for _ in range(2):
            e = task.generate_example()
            lines.append("### Example")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(task.render_prompt(e.metadata))
            lines.append("```")
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append("```")
            lines.append(e.answer)
            lines.append("```")
            lines.append("")
        lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
