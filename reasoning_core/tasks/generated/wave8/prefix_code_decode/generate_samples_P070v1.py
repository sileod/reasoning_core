import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.prefix_code_decode.prefix_code_decode import PrefixCodeDecode

SEED = 1934978647
OUT = Path(__file__).with_name("samples_P070v1.md")


def main():
    random.seed(SEED)
    task = PrefixCodeDecode()
    lines = ["# Prefix Code Decode v1 samples", ""]
    for level, count in ((0, 2), (2, 2), (5, 2)):
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(count):
            e = task.generate_example()
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
