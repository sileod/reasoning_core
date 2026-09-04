import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.markov_chain_distribution.markov_chain_distribution import (
    MarkovChainDistribution,
)


def main():
    random.seed(2267388306)
    task = MarkovChainDistribution()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"Level {level}")
        for _ in range(2):
            task.config.set_level(level)
            ex = task.generate_example()
            lines.append(f"Prompt:\n{task.prompt(ex.metadata)}")
            lines.append(f"Answer: {ex.answer}")
    out = Path(__file__).with_name("samples_P003v1.md")
    out.write_text("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
