import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.paxos_chosen_value.paxos_chosen_value import (
    PaxosChosenValue, PaxosChosenValueConfig
)

random.seed(2325212251)

out = Path(__file__).with_name("samples_P041v1.md")

lines = []
lines.append("# Samples P041v1: paxos_chosen_value")
lines.append("")

for level in (0, 2, 5):
    lines.append(f"## Level {level}")
    lines.append("")
    for i in range(2):
        cfg = PaxosChosenValueConfig()
        cfg.apply_difficulty(level)
        t = PaxosChosenValue()
        t.config = cfg
        e = t.generate_example()
        lines.append("### Example")
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append(e.metadata.prompt if "prompt" in e.metadata else t.render_prompt(e.metadata))
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(e.answer)
        lines.append("")

out.write_text("\n".join(lines))
print("wrote", out)
