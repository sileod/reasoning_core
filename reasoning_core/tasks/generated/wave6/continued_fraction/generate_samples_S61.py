import random
from pathlib import Path

from reasoning_core.tasks.generated.wave6.s61_continued_fraction.s61_continued_fraction import ContinuedFraction, ContinuedFractionConfig

SEED = 1513341689


def main():
    random.seed(SEED)
    out = []
    for level in [0, 2, 5]:
        out.append(f"### Level {level}")
        config = ContinuedFractionConfig()
        config.set_level(level)
        task = ContinuedFraction(config=config)
        for _ in range(2):
            entry = task.generate_example()
            out.append("Prompt: " + task.render_prompt(entry.metadata))
            out.append("Answer: " + entry.answer)
            out.append("")
    text = "\n".join(out)
    path = Path(__file__).with_name("samples_S61.md")
    path.write_text(text)


if __name__ == "__main__":
    main()
