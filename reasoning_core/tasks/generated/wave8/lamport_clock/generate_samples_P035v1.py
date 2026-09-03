import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.lamport_clock.lamport_clock import LamportClock


def main():
    random.seed(2791137760)
    task = LamportClock()
    task.config.seed = 2791137760
    out = Path(__file__).with_name("samples_P035v1.md")
    blocks = []
    for level in (0, 2, 5):
        blocks.append(f"## Level {level}")
        for _ in range(2):
            ex = task.generate_example(level=level)
            print("--------------------------------------------------------------")
            print("PROMPT:")
            print(ex.metadata._prompt if hasattr(ex.metadata, "_prompt") else task.render_prompt(ex.metadata))
            print("ANSWER:")
            print(ex.answer)
            blocks.append(f"### Example {_ + 1}")
            blocks.append("Prompt:")
            blocks.append("```")
            blocks.append(task.render_prompt(ex.metadata))
            blocks.append("```")
            blocks.append("Answer:")
            blocks.append(f"```\n{ex.answer}\n```")
        blocks.append("")
    out.write_text("\n".join(blocks))
    print("wrote", out)


if __name__ == "__main__":
    main()
