import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.abstract_interval_analysis.abstract_interval_analysis import (
    AbstractIntervalAnalysis,
)

SEED = 1549547534
OUT = Path(__file__).resolve().parent / "samples_P030v1.md"


def main():
    random.seed(SEED)
    t = AbstractIntervalAnalysis()
    lines = []
    for L in (0, 2, 5):
        lines.append(f"# Level {L}\n")
        t.config.set_level(L)
        for i in range(2):
            e = t.generate_example()
            lines.append(f"## Example {i + 1}\n")
            lines.append("### Prompt\n")
            lines.append(t.render_prompt(e.metadata) + "\n")
            lines.append("### Answer\n")
            lines.append(e.answer + "\n")
        lines.append("\n")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
