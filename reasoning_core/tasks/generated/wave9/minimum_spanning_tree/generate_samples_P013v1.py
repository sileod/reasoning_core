from pathlib import Path

import random

from reasoning_core.tasks.generated.wave9.minimum_spanning_tree.minimum_spanning_tree import (
    MinimumSpanningTree,
)

OUT = Path(__file__).with_name("samples_P013v1.md")


def main():
    random.seed(3901837384)
    task = MinimumSpanningTree()
    levels = {0, 2, 5}
    with open(OUT, "w") as f:
        for level in sorted(levels):
            f.write(f"## Level {level}\n\n")
            cfg = task.config_cls()
            cfg.set_level(level)
            task.config = cfg
            for i in range(2):
                e = task.generate_example()
                f.write(f"### Example {i + 1}\n\n")
                f.write("**Prompt:**\n\n```\n")
                f.write(task.render_prompt(e.metadata))
                f.write("```\n\n")
                f.write(f"**Answer:** {e.answer}\n\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
