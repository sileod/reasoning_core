import random
from pathlib import Path

SEED = 69737855


def generate_samples():
    from reasoning_core.tasks.generated.wave10.event_order_reconstruction.event_order_reconstruction import (
        EventOrderReconstruction,
    )

    task = EventOrderReconstruction()
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append("## Level %d" % level)
        for i in range(2):
            random.seed(SEED + level * 100 + i)
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            out.append("")
            out.append("### Example %d" % (i + 1))
            out.append("")
            out.append("**Prompt:**")
            out.append(prompt)
            out.append("")
            out.append("**Answer:**")
            out.append(x.answer)
        out.append("")

    path = Path(__file__).with_name("samples_P017v1.md")
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    generate_samples()
