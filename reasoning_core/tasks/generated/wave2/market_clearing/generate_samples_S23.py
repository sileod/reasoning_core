import os
import random

from reasoning_core.tasks.generated.wave2.s23_market_clearing.market_clearing import MarketClearing, MarketClearingConfig


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    random.seed(2836545685)
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}\n")
        task = MarketClearing(config=MarketClearingConfig(), level=level)
        for i in range(2):
            x = task.generate_example()
            lines.append(f"### Example {i + 1}\n")
            lines.append(f"**Prompt:**\n\n```\n{task.render_prompt(x.metadata)}\n```\n")
            lines.append(f"**Answer:** {x.answer}\n")
        lines.append("")
    with open(os.path.join(here, "samples_S23.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
