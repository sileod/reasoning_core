from pathlib import Path
import random

from reasoning_core.tasks.generated.wave8.generated_subgroup_membership.generated_subgroup_membership import (
    GeneratedSubgroupMembership,
)


def main():
    random.seed(710918460)
    out = Path(__file__).with_name("samples_P021v1.md")
    lines = []
    for level in (0, 2, 5):
        t = GeneratedSubgroupMembership()
        t.config.set_level(level)
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            ex = t.generate_example()
            lines.append("Prompt:")
            lines.append("")
            lines.append(t.render_prompt(ex.metadata))
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
