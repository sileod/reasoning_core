import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s46_voting_rules.voting_rules import VotingRules


def main():
    random.seed(736651786)
    out = Path(__file__).with_name("samples_S46.md")
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task = VotingRules()
        task.config.set_level(level)
        for i in range(2):
            x = task.generate_example()
            lines.append(f"### Example {i+1}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(x.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(f"`{x.answer}`")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
