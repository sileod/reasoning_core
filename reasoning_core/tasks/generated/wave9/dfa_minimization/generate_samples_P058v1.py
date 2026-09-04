import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.dfa_minimization.dfa_minimization import DfaMinimization

random.seed(1633967595)

task = DfaMinimization()
lines = []
for level in (0, 2, 5):
    task.config.set_level(level)
    lines.append(f"# Level {level}")
    for i in range(2):
        x = task.generate_example()
        lines.append(f"## Example {i+1}")
        lines.append("### Prompt")
        lines.append(task.render_prompt(x.metadata))
        lines.append("### Answer")
        lines.append(x.answer)
    lines.append("")

out = Path(__file__).with_name("samples_P058v1.md")
out.write_text("\n".join(lines))
print(out)
