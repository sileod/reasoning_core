import os
import random

from reasoning_core.tasks.generated.wave3.s32_lexicographic_ranking.lexicographic_ranking import (
    LexicographicRanking,
)


def main():
    random.seed(1922532027)
    task = LexicographicRanking()
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for level in [0, 2, 5]:
        task.config.set_level(level)
        out.append("## Level " + str(level))
        for i in range(3):
            ex = task.generate_example()
            out.append("### Example " + str(i + 1))
            out.append("Prompt:")
            out.append(task.render_prompt(ex.metadata))
            out.append("")
            out.append("Answer:")
            out.append(ex.answer)
            out.append("")
    with open(os.path.join(here, "samples_S32.md"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
