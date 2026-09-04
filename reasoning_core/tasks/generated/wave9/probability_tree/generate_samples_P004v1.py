from pathlib import Path
import random

from reasoning_core.tasks.generated.wave9.probability_tree_marginal.probability_tree_marginal import ProbabilityTree

SEED = 3536382515
OUT = Path(__file__).with_name("samples_P004v1.md")


def main():
    random.seed(SEED)
    lines = ["# Samples for probability_tree_marginal (P004v1)", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        obj = ProbabilityTree()
        obj.config.set_level(level)
        for i in range(2):
            entry = obj.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append("")
            lines.append(obj.render_prompt(entry.metadata))
            lines.append("")
            lines.append(f"**Answer:** {entry.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
