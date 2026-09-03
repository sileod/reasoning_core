import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.approval_voting_winner.approval_voting_winner import ApprovalVotingWinner

random.seed(3147129343)

task = ApprovalVotingWinner()
out = Path(__file__).with_name("samples_P065v1.md")
lines = []
for level in (0, 2, 5):
    lines.append(f"# Level {level}")
    lines.append("")
    for _ in range(2):
        x = task.generate_example(level=level)
        lines.append("**Prompt:**")
        lines.append("```")
        lines.append(x.prompt)
        lines.append("```")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("```")
        lines.append(x.answer)
        lines.append("```")
        lines.append("")

out.write_text("\n".join(lines) + "\n")
print(out)
