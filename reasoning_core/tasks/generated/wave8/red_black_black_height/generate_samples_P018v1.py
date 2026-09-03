import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.red_black_black_height.red_black_black_height import (
    RedBlackBlackHeight,
)


random.seed(306550574)


def make_example(level):
    task = RedBlackBlackHeight()
    cfg = task.config
    cfg.set_level(level)
    task.config = cfg
    x = task.generate_example()
    prompt = task.render_prompt(x.metadata)
    return prompt, x.answer


def main():
    samples = []
    for level in (0, 2, 5):
        samples.append(f"# Level {level}\n")
        for _ in range(2):
            prompt, answer = make_example(level)
            samples.append("Prompt:\n")
            samples.append(prompt + "\n")
            samples.append("Answer:\n")
            samples.append(answer + "\n")

    out = Path(__file__).with_name("samples_P018v1.md")
    out.write_text("\n".join(samples))


if __name__ == "__main__":
    main()
