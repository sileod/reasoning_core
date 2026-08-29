import random
from pathlib import Path

from reasoning_core.tasks.mutated.wave0.m02_regex_boolean_languages.m02_regex_boolean_languages import (
    RegexBooleanLanguages,
)

random.seed(117786424)

OUT = Path(__file__).parent / "samples_M2.md"

LINES = []
for level in (0, 2, 5):
    task = RegexBooleanLanguages()
    task.config.set_level(level)
    for _ in range(2):
        entry = task.generate_entry()
        LINES.append(f"\nLevel {level}\n")
        LINES.append("Prompt:\n")
        LINES.append(task.render_prompt(entry.metadata))
        LINES.append("\nAnswer:\n")
        LINES.append(entry.answer)
        LINES.append("\n")

OUT.write_text("\n".join(LINES))
print("wrote", OUT)
