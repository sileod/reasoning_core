import random
from pathlib import Path

from reasoning_core.template import Task
from reasoning_core.tasks.generated.wave9.two_phase_commit_trace.two_phase_commit_trace import (
    TwoPhaseCommitTrace,
    TwoPhaseCommitConfig,
)

SEED = 2867049114
OUT = Path(__file__).with_name("samples_P078v1.md")


def main():
    random.seed(SEED)
    task = TwoPhaseCommitTrace()
    lines = []
    lines.append("# Two-phase commit trace: samples_P078v1")
    lines.append("")
    for level in (0, 2, 5):
        cfg = TwoPhaseCommitConfig()
        cfg.set_level(level)
        task.config = cfg
        lines.append("## Level %d" % level)
        lines.append("")
        for i in range(2):
            ex = task.generate_example()
            prompt = task.render_prompt(ex.metadata)
            lines.append("### Example %d" % (i + 1))
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(prompt)
            lines.append("")
            lines.append("**Answer:** `%s`" % ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
