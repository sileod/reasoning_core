import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import random
from reasoning_core.tasks.generated.wave9.relational_join_execution.relational_join import RelationalJoin

random.seed(2738823173)

task = RelationalJoin()

out_path = Path(__file__).with_name("samples_P031v1.md")
lines = []
for level in (0, 2, 5):
    task.config.set_level(level)
    lines.append(f"## Level {level}")
    for _ in range(2):
        ex = task.generate_example()
        lines.append("### Example")
        lines.append("#### Prompt")
        lines.append(task.render_prompt(ex.metadata))
        lines.append("#### Answer")
        lines.append(ex.answer)
        lines.append("")

out_path.write_text("\n".join(lines))
print(out_path)
