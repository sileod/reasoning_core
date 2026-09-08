import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.novel_word_application.novel_word_application import NovelWordApplication, NovelWordApplicationConfig


def main():
    random.seed(1621874809)
    out = Path(__file__).with_name("samples_P013v1.md")
    lines = []
    for level in (0, 2, 5):
        cfg = NovelWordApplicationConfig()
        cfg.apply_difficulty(level)
        task = NovelWordApplication(config_cls=NovelWordApplicationConfig)
        task.config = cfg
        lines.append(f"## Level {level}")
        for _ in range(2):
            e = task.generate_example()
            lines.append("### Example")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
