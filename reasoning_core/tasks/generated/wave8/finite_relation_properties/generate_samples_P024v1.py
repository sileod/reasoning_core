import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.finite_relation_properties.finite_relation_properties import (
    FiniteRelationProperties,
)

random.seed(2261582844)


def main():
    task = FiniteRelationProperties()
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"# Level {level}\n")
        for _ in range(2):
            e = task.generate_example()
            out.append("**Prompt**\n\n")
            out.append(task.render_prompt(e.metadata) + "\n\n")
            out.append("**Answer**\n\n")
            out.append(e.answer + "\n\n")
            out.append("---\n\n")
    path = Path(__file__).with_name("samples_P024v1.md")
    path.write_text("".join(out))


if __name__ == "__main__":
    main()
