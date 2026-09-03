import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

random.seed(991003721)

from reasoning_core.tasks.generated.wave8.lattice_join_meet.lattice_join_meet import (
    LatticeJoinMeet,
    LatticeJoinMeetConfig,
)

OUT = Path(__file__).with_name("samples_P026v1.md")


def main():
    task = LatticeJoinMeet()
    lines = []
    for level in (0, 2, 5):
        cfg = LatticeJoinMeetConfig()
        cfg.set_level(level)
        task.config = cfg
        lines.append(f"## Level {level}")
        lines.append("")
        for k in range(2):
            ex = task.generate_example()
            lines.append(f"### Example {k + 1}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
