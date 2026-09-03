import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.raft_vote_eligibility.raft_vote_eligibility import (
    RaftVoteEligibility,
)

SEED = 2471421591

LEVELS = [0, 2, 5]
EXAMPLES_PER_LEVEL = 2


def main():
    random.seed(SEED)
    task = RaftVoteEligibility()
    out = Path(__file__).with_name("samples_P042v1.md")
    lines = []
    for level in LEVELS:
        task.config.set_level(level)
        lines.append(f"\n## Level {level}\n")
        for _ in range(EXAMPLES_PER_LEVEL):
            x = task.generate_example()
            lines.append("### Prompt\n")
            lines.append(task.render_prompt(x.metadata))
            lines.append("\n### Answer\n")
            lines.append(x.answer)
            lines.append("\n")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
