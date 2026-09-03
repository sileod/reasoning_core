import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.set_associative_cache.set_associative_cache import (
    SetAssociativeCache,
    SetAssociativeCacheConfig,
)


def main():
    random.seed(1952221362)
    out = Path(__file__).with_name("samples_P044v1.md")
    lines = []
    levels = [0, 2, 5]
    for level in levels:
        lines.append(f"## Level {level}")
        cfg = SetAssociativeCacheConfig()
        cfg.set_level(level)
        task = SetAssociativeCache(config=cfg)
        for i in range(2):
            entry = task.generate_example()
            lines.append(f"### Example {i + 1}")
            lines.append(task.render_prompt(entry.metadata))
            lines.append("")
            lines.append("Answer:")
            lines.append(entry.answer)
            lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
