import os
import random
from reasoning_core.template import Task

from signed_digit_arithmetic import SignedDigitArithmetic


def main():
    random.seed(3574012244)
    task = SignedDigitArithmetic()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "samples_S33.md"), "w") as f:
        for level in (0, 2, 5):
            task.config.set_level(level)
            f.write(f"## Level {level}\n\n")
            for _ in range(2):
                entry = task.generate_example()
                f.write("### Prompt\n\n")
                f.write(task.render_prompt(entry.metadata))
                f.write("\n\n### Answer\n\n")
                f.write(entry.answer)
                f.write("\n\n")


if __name__ == "__main__":
    main()
