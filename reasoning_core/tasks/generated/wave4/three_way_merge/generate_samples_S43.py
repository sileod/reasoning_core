import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s43_three_way_merge.three_way_merge import (
    ThreeWayMerge,
)

SEED = 3493938574
LEVELS = {0: 2, 2: 2, 5: 2}


def main():
    random.seed(SEED)
    out = []
    out.append("# Samples for s43_three_way_merge")
    out.append("")
    t = ThreeWayMerge()
    for level, count in LEVELS.items():
        out.append("## Level %d" % level)
        out.append("")
        t.config.set_level(level)
        for _ in range(count):
            e = t.generate_example()
            prompt = t.render_prompt(e.metadata)
            out.append("### Prompt")
            out.append("")
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append("")
            out.append("**Answer**")
            out.append("")
            out.append("```")
            out.append(e.answer)
            out.append("```")
            out.append("")
    path = Path(__file__).with_name("samples_S43.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
