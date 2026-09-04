import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.articulation_bridges.articulation_bridges import ArticulationBridges


def main():
    random.seed(1881068873)
    task = ArticulationBridges()
    out = Path(__file__).with_name("samples_P016v1.md")
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"# Level {level}")
        lines.append("")
        for _ in range(2):
            ex = task.generate_example()
            lines.append("Prompt:")
            lines.append(ex.prompt)
            lines.append("")
            lines.append("Answer:")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
