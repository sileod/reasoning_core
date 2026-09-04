import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.expected_utility_choice.expected_utility_choice import (
    ExpectedUtilityChoice,
    ExpectedUtilityConfig,
)


def main():
    random.seed(798610012)
    task = ExpectedUtilityChoice()
    out = []
    for level in (0, 2, 5):
        cfg = ExpectedUtilityConfig()
        cfg.set_level(level)
        task.config = cfg
        out.append(f"## Level {level}\n")
        for _ in range(2):
            entry = task.generate_example()
            out.append("### Prompt")
            out.append("```")
            out.append(task.render_prompt(entry.metadata))
            out.append("```")
            out.append("### Answer")
            out.append("```")
            out.append(entry.answer)
            out.append("```")
            out.append("")
    path = Path(__file__).with_name("samples_P006v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
