import random
from pathlib import Path

from reasoning_core.tasks.generated.wave3.s31_grundy_values.grundy_values import GrundyValues

random.seed(3693162845)

OUT = Path(__file__).resolve().parent / "samples_S31.md"


def render_example(task, level):
    task.config.seed = 3693162845
    ex = task.generate_example(level=level)
    prompt = task.render_prompt(ex.metadata)
    return prompt, ex.answer


def main():
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task = GrundyValues()
        for _ in range(2):
            prompt, answer = render_example(task, level)
            lines.append("Prompt:")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("")
            lines.append(f"Answer: {answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
