import random

seed = 747576363
random.seed(seed)

from reasoning_core.tasks.generated.wave1.s10_heap_simulation.heap_simulation import HeapSimulation, HeapSimulationConfig  # noqa: E402


def emit(level):
    cfg = HeapSimulationConfig()
    cfg.set_level(level)
    task = HeapSimulation()
    task.config = cfg
    lines = [f"### Level {level}"]
    for _ in range(2):
        x = task.generate_example()
        lines.append(random.choices(["\n", "\n\n"])[0])
        lines.append("Prompt:")
        lines.append(task.render_prompt(x.metadata))
        lines.append("\nAnswer:")
        lines.append(x.answer)
    return "\n".join(lines)


parts = []
for lvl in (0, 2, 5):
    parts.append(emit(lvl))

import os

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "samples_S10.md"), "w") as f:
    f.write("\n".join(parts) + "\n")
