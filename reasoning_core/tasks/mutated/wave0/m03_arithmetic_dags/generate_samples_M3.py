import random
import os
from reasoning_core.tasks.mutated.wave0.m03_arithmetic_dags.dag_arithmetics import DagArithmetics

SEED = 2588473867
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples_M3.md")


def render(eg, task):
    return (
        f"### Level {task.config.level} example\n\n"
        f"**Prompt:**\n\n{task.render_prompt(eg.metadata)}\n\n"
        f"**Answer:** `{eg.answer}`\n"
    )


def main():
    random.seed(SEED)
    with open(OUT, "w") as f:
        for level in (0, 0, 2, 2, 5, 5):
            task = DagArithmetics()
            task.config.set_level(level)
            eg = task.generate_example()
            f.write(f"## Level {level}\n\n")
            f.write(render(eg, task))
            f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
