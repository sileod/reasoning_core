import random

from reasoning_core.tasks.generated.wave3.s30_resistor_networks.s30_resistor_networks import (
    ResistorNetworks,
    ResistorNetworkConfig,
)

random.seed(564967272)

sections = {0: [], 2: [], 5: []}
for level in sorted(sections):
    config = ResistorNetworkConfig()
    config.set_level(level)
    task = ResistorNetworks()
    task.config = config
    for _ in range(2):
        entry = task.generate_example()
        sections[level].append(entry)

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples_S30.md")
with open(out_path, "w") as f:
    for level, entries in sorted(sections.items()):
        f.write(f"## Level {level}\n\n")
        for i, entry in enumerate(entries, 1):
            f.write(f"### Example {i}\n\n")
            f.write(task.render_prompt(entry.metadata) + "\n\n")
            f.write(f"**Answer:** {entry.answer}\n\n")
