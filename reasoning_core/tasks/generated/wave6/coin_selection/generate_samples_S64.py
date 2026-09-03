import random
from pathlib import Path

from reasoning_core.tasks.generated.wave6.s64_coin_selection.coin_selection import CoinSelection


def main():
    random.seed(241888686)
    task = CoinSelection()
    out = Path(__file__).with_name("samples_S64.md")
    lines = ["# Samples for S64 coin_selection", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append("")
            lines.append("Prompt:")
            lines.append("")
            lines.append("```")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("```")
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append("```")
            lines.append(ex.answer)
            lines.append("```")
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
