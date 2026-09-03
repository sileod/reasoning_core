import random

from reasoning_core.template import stochastic_rounding as sround


def write_samples(path):
    from reasoning_core.tasks.generated.wave8.dimensional_consistency.dimensional_consistency import (
        DimensionalConsistency,
    )

    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        task = DimensionalConsistency()
        try:
            task.config.set_level(level)
        except Exception:
            pass
        # Generate two examples
        shown = 0
        guard = 0
        while shown < 2 and guard < 200:
            guard += 1
            ex = task.generate_example()
            prompt = task.render_prompt(ex.metadata)
            if not prompt.strip():
                continue
            lines.append(f"\n### Example {shown + 1}")
            lines.append(f"\n**Prompt:**")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append(f"**Answer:**")
            lines.append("")
            lines.append(f"`{ex.answer}`")
            lines.append("")
            shown += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    random.seed(3968631907)
    out = Path(__file__).with_name("samples_P073v2.md")
    write_samples(out)
    print(out)
