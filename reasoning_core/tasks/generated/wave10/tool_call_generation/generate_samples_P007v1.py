import random

from pathlib import Path

from reasoning_core.tasks.generated.wave10.tool_call_generation.tool_call_generation import (
    ToolCallGeneration,
)

OUT = Path(__file__).with_name("samples_P007v1.md")


def main():
    random.seed(1351444575)
    task = ToolCallGeneration()
    with open(OUT, "w") as f:
        for level in (0, 2, 5):
            f.write(f"## Level {level}\n\n")
            for _ in range(2):
                ex = task.generate_example(level=level)
                f.write("**Prompt:**\n\n")
                f.write(ex.prompt + "\n\n")
                f.write("**Answer:**\n\n")
                f.write(ex.answer + "\n\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
