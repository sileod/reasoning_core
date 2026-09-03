import random
from pathlib import Path

from reasoning_core.tasks.generated.wave5.s57_shipping_route.s57_shipping_route import ShippingRoute

random.seed(3982785801)

OUT = Path(__file__).with_name("samples_S57.md")


def render_level(level, n):
    task = ShippingRoute()
    task.config.set_level(level)
    lines = ["## Level %d" % level, ""]
    for _ in range(n):
        e = task.generate_example()
        lines.append("Prompt:")
        lines.append(task.render_prompt(e.metadata))
        lines.append("")
        lines.append("Answer:")
        lines.append(e.answer)
        lines.append("")
    return "\n".join(lines)


def main():
    parts = []
    for level in (0, 2, 5):
        parts.append(render_level(level, 2))
    OUT.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
