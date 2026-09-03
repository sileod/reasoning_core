import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.pattern_match_exhaustiveness.pattern_match_exhaustiveness import (
    PatternMatchExhaustiveness,
)

SEED = 1344474452
OUT = Path(__file__).with_name("samples_P060v1.md")


def _example(task, level):
    task.config.set_level(level)
    entry = task.generate_example()
    prompt = task.render_prompt(entry.metadata)
    return prompt, entry.answer


def main():
    random.seed(SEED)
    task = PatternMatchExhaustiveness()
    with open(OUT, "w") as f:
        f.write("# samples_P060v1\n\n")
        f.write("Pattern-match exhaustiveness: decide whether all constructors are covered,\n")
        f.write("naming the smallest uncovered constructor as a counterexample.\n\n")
        for level in (0, 2, 5):
            f.write(f"## Level {level}\n\n")
            for i in range(2):
                prompt, answer = _example(task, level)
                f.write(f"### Example {i+1}\n\n")
                f.write(prompt)
                f.write("\n\n**Answer:** ")
                f.write(answer)
                f.write("\n\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
