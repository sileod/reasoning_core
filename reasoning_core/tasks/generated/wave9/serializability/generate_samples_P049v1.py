import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.transaction_serializability.transaction_serializability import (
    Serializability,
)


def main():
    random.seed(3323899505)
    task = Serializability()
    out = []
    for level in (0, 2, 5):
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        out.append("## Level {}".format(level))
        for _ in range(2):
            entry = task.generate_entry()
            out.append("")
            out.append("### Prompt")
            out.append("```")
            out.append(task.render_prompt(entry.metadata))
            out.append("```")
            out.append("")
            out.append("### Answer")
            out.append("```")
            out.append(entry.answer)
            out.append("```")
            out.append("")
    path = Path(__file__).with_name("samples_P049v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
