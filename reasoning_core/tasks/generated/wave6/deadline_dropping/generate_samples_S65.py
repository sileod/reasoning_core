import random
from pathlib import Path
from reasoning_core.tasks.generated.wave6.s65_deadline_dropping.s65_deadline_dropping import DeadlineDropping, DeadlineDroppingConfig

random.seed(65042966)

levels = [0, 2, 5]
out = []
task = DeadlineDropping()
for level in levels:
    cfg = DeadlineDroppingConfig()
    cfg.set_level(level)
    task.config = cfg
    out.append(f"## Level {level}\n")
    for _ in range(2):
        ex = task.generate_example()
        out.append("### Prompt\n")
        out.append(task.render_prompt(ex.metadata))
        out.append("\n")
        out.append("### Answer\n")
        out.append(ex.answer)
        out.append("\n")

Path(__file__).with_name("samples_S65.md").write_text("\n".join(out))
