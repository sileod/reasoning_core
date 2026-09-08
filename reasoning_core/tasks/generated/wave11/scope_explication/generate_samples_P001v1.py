import random
from pathlib import Path

from reasoning_core.template import Task
from reasoning_core.tasks.generated.wave11.scope_explication.scope_explication import (
    ScopeExplication, ScopeExplicationConfig,
)

SEED = 974528455
OUT = Path(__file__).with_name("samples_P001v1.md")

LEVELS = {0: 2, 2: 2, 5: 2}


def main():
    random.seed(SEED)
    task = ScopeExplication(config=ScopeExplicationConfig())
    lines = ["# samples_P001v1",
             "",
             "Seed: %d. Two complete prompt/answer examples per level 0, 2 and 5." % SEED]
    for level, n in LEVELS.items():
        lines.append("")
        lines.append("## Level %d" % level)
        for i in range(n):
            ex = task.generate_example(level=level)
            lines.append("")
            lines.append("### Prompt %d" % (i + 1))
            lines.append("")
            lines.append(ex.prompt)
            lines.append("")
            lines.append("### Answer %d" % (i + 1))
            lines.append("")
            lines.append(ex.answer)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
