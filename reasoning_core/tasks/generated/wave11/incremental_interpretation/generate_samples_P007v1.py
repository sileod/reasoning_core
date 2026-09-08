import random
from pathlib import Path

from reasoning_core.tasks.generated.wave11.incremental_interpretation.incremental_interpretation import (
    IncrementalInterpretation,
)

OUT = Path(__file__).with_name("samples_P007v1.md")

LEVELS = [0, 2, 5]

def main():
    random.seed(584479410)
    lines = []
    for lvl in LEVELS:
        lines.append(f"# Level {lvl}")
        task = IncrementalInterpretation()
        for i in range(2):
            e = task.generate_example(level=lvl)
            prompt = task.render_prompt(e.metadata)
            lines.append("## Example")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
