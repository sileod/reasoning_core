import random
from pathlib import Path

random.seed(4082635274)

from reasoning_core.tasks.generated.wave8.incremental_build_rebuild_set.incremental_build_rebuild_set import (
    RebuildSet,
    RebuildSetConfig,
)


def main():
    out = Path(__file__).with_name("samples_P079v1.md")
    task = RebuildSet()
    lines = []
    for level in (0, 2, 5):
        cfg = RebuildSetConfig()
        cfg.set_level(level)
        task.config = cfg
        lines.append(f"# Level {level}\n")
        for _ in range(2):
            e = task.generate_example()
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("Answer: " + e.answer)
            lines.append("")
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
