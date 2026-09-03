import os
import random

from reasoning_core.tasks.generated.wave3.s27_term_unification.term_unification import TermUnification


def main():
    random.seed(1680339059)
    out = []
    for level in [0, 2, 5]:
        task = TermUnification()
        task.config.set_level(level)
        out.append("## Level %d\n" % level)
        for _ in range(2):
            e = task.generate_example()
            out.append(task.render_prompt(e.metadata))
            out.append("")
            out.append("Answer: %s" % e.answer)
            out.append("")
    text = "\n".join(out)
    here = os.path.dirname(os.path.abspath(__file__))
    trial = "S27"
    path = os.path.join(here, "samples_%s.md" % trial)
    with open(path, "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    main()
