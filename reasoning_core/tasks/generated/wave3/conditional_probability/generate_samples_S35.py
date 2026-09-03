import random
from pathlib import Path

from reasoning_core.tasks.generated.wave3.s35_conditional_probability.conditional_probability import (
    ConditionalProbability,
)


def main():
    random.seed(1225289588)
    t = ConditionalProbability()
    out = []
    for level in (0, 2, 5):
        t.config.set_level(level)
        out.append("## Level %d\n" % level)
        for i in range(2):
            ex = t.generate_example()
            out.append("### Example %d\n" % (i + 1))
            out.append("**Prompt:**\n\n%s\n" % ex.metadata.payload["question"])
            out.append("**Answer:**\n\n%s\n" % ex.answer)
        out.append("\n")
    path = Path(__file__).with_name("samples_S35.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
