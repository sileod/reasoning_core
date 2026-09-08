import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.information_seeking.information_seeking import (
    InfoSeeking,
    InfoSeekingConfig,
)


def main():
    random.seed(147658999)
    out = Path(__file__).with_name("samples_P019v1.md")
    task = InfoSeeking()
    blocks = []
    for lvl in (0, 2, 5):
        cfg = InfoSeekingConfig()
        cfg.set_level(lvl)
        task.config = cfg
        blocks.append(f"## Level {lvl}\n")
        for _ in range(2):
            ex = task.generate_example()
            prompt = task.render_prompt(ex.metadata)
            blocks.append("### Prompt\n")
            blocks.append("```\n" + prompt + "\n```\n")
            blocks.append("### Answer\n")
            blocks.append("```\n" + ex.answer + "\n```\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
