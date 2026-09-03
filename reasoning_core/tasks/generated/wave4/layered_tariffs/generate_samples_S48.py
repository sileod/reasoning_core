import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s48_layered_tariffs.s48_layered_tariffs import (
    LayeredTariffs,
)


def main():
    random.seed(709414788)
    t = LayeredTariffs()
    out = []
    for level in (0, 2, 5):
        cfg = t.config_cls()
        cfg.set_level(level)
        t.config = cfg
        out.append(f"## Level {level}\n")
        for i in range(2):
            x = t.generate_example()
            out.append(f"### Example {i + 1}\n")
            out.append("**Prompt:**\n")
            out.append(t.render_prompt(x.metadata))
            out.append("\n**Answer:**\n")
            out.append(x.answer)
            out.append("\n")
        out.append("\n")
    path = Path(__file__).with_name("samples_S48.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
