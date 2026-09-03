import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.petri_enabled_transitions.petri_enabled_transitions import (
    PetriEnabledTransitions,
)


def main():
    random.seed(1701447588)
    out_path = Path(__file__).with_name("samples_P077v1.md")
    task = PetriEnabledTransitions()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        for _ in range(2):
            e = task.generate_example()
            prompt = task.render_prompt(e.metadata)
            lines.append(prompt)
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
