"""Generate samples_P066v1.md: two complete prompt/answer examples at levels 0, 2, 5."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.top_trading_cycles.top_trading_cycles import TopTradingCycles

TRIAL = "P066v1"
SEED = 623615645
OUT = Path(__file__).with_name(f"samples_{TRIAL}.md")


def main():
    random.seed(SEED)
    task = TopTradingCycles()
    lines = [f"# Samples {TRIAL}", ""]
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        for _ in range(2):
            e = task.generate_example(level=level)
            lines.append("### Example")
            lines.append("")
            lines.append("**Prompt**")
            lines.append("")
            lines.append(task.render_prompt(e.metadata))
            lines.append("")
            lines.append("**Answer**")
            lines.append("")
            lines.append(e.answer)
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
