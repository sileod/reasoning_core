import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.candidate_key_minimality.candidate_key_minimality import (
    CandidateKeyMinimality,
)

SEED = 3142999616
OUT = Path(__file__).with_name("samples_P028v1.md")


def run():
    random.seed(SEED)
    task = CandidateKeyMinimality()
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append("## Level %d" % level)
        lines.append("")
        for _ in range(2):
            ex = task.generate_example()
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("```")
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    run()
