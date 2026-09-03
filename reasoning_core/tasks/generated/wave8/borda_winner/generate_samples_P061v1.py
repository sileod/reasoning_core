"""Generate samples_P061v1.md for the Borda winner task."""
import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.borda_winner.borda_winner import BordaWinner

SEED = 2916049189
OUT = Path(__file__).with_name("samples_P061v1.md")


def main():
    random.seed(SEED)
    task = BordaWinner()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task.config.set_level(level)
        for _ in range(2):
            x = task.generate_example()
            lines.append("**Prompt:**")
            lines.append("```")
            lines.append(task.render_prompt(x.metadata))
            lines.append("```")
            lines.append("**Answer:**")
            lines.append("```")
            lines.append(x.answer)
            lines.append("```")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
