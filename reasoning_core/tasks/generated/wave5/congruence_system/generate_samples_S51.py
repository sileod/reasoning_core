import random
from pathlib import Path

from s51_congruence_system import CongruenceSystem


def main():
    random.seed(2311544729)
    out = Path(__file__).with_name("samples_S51.md")
    t = CongruenceSystem()
    lines = []
    for level in (0, 2, 5):
        t.config.set_level(level)
        lines.append("## Level %d" % level)
        for _ in range(2):
            e = t.generate_example()
            lines.append(t.render_prompt(e.metadata))
            lines.append("")
            lines.append("Answer: %s" % e.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
