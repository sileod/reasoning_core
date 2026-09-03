"""Generate samples_P029v2.md for the bcnf_violation trial.

Seeded with 4052403849 for byte-reproducible output.
"""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.bcnf_violation.bcnf_violation import BcnfViolation

OUT = Path(__file__).with_name("samples_P029v2.md")
SEED = 4052403849


def main():
    random.seed(SEED)
    task = BcnfViolation()
    lines = []
    lines.append("# BCNF violation v2 samples")
    lines.append("")
    for level in (0, 2, 5):
        cfg = BcnfViolation.config_cls()
        cfg.set_level(level)
        task.config = cfg
        lines.append(f"## Level {level}")
        lines.append("")
        for i in range(2):
            entry = task.generate_entry()
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append("> " + task.render_prompt(entry.metadata).replace("\n", "\n> "))
            lines.append("")
            lines.append(f"Answer: {entry.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
