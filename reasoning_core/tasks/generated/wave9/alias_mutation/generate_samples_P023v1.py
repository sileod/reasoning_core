import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.alias_mutation_tracking.alias_mutation_tracking import AliasMutation


def main():
    random.seed(1337854016)
    task = AliasMutation()
    out = []
    for level in (0, 2, 5):
        out.append(f"## Level {level}")
        out.append("")
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        for i in range(2):
            e = task.generate_example()
            out.append(f"### Example {i + 1}")
            out.append("")
            out.append("Prompt:")
            out.append("")
            out.append(task.render_prompt(e.metadata))
            out.append("")
            out.append("Answer:")
            out.append("")
            out.append(f"```\n{e.answer}\n```")
            out.append("")
    (Path(__file__).with_name("samples_P023v1.md")).write_text("\n".join(out))


if __name__ == "__main__":
    main()
