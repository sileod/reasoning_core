import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.round_robin_completion.round_robin_completion import (
    RoundRobinCompletion,
)

random.seed(617708580)


def gen(level):
    t = RoundRobinCompletion()
    t.config.set_level(level)
    return t


def section(level):
    out = [f"## Level {level}"]
    t = gen(level)
    for _ in range(2):
        x = t.generate_example()
        out.append("**Prompt:**")
        out.append(t.render_prompt(x.metadata))
        out.append("")
        out.append("**Answer:**")
        out.append(x.answer)
        out.append("")
    return "\n".join(out)


def main():
    parts = ["# Round Robin Completion samples (P050v1)\n"]
    for level in (0, 2, 5):
        parts.append(section(level))
    content = "\n".join(parts) + "\n"
    path = Path(__file__).with_name("samples_P050v1.md")
    path.write_text(content)
    print("wrote", path)


if __name__ == "__main__":
    main()
