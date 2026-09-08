"""Generate samples_P011v2.md for the agreement_completion trial.

Seeded so the output is byte-reproducible across processes and runs.
"""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.agreement_completion.agreement_completion import (
    AgreementCompletion,
)

SEED = 2738521636
LEVELS = ("0", "2", "5")
PER_LEVEL = 2

random.seed(SEED)
task = AgreementCompletion()

out = ["# agreement_completion samples", ""]
out.append("Task: supply the correctly inflected next word despite intervening "
           "nouns, coordination, and nested relative clauses.")
out.append("")

for level in LEVELS:
    task.config.set_level(int(level))
    out.append(f"## Level {level}")
    out.append("")
    for _ in range(PER_LEVEL):
        entry = task.generate_example()
        prompt = task.render_prompt(entry.metadata)
        out.append("Prompt:")
        out.append("```")
        out.append(prompt)
        out.append("```")
        out.append("")
        out.append("Answer:")
        out.append("```")
        out.append(entry.answer)
        out.append("```")
        out.append("")

path = Path(__file__).with_name("samples_P011v2.md")
path.write_text("\n".join(out))
print(f"wrote {path}")
