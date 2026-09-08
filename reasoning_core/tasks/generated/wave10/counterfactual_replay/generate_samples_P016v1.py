import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.counterfactual_replay.counterfactual_replay import (
    CounterfactualReplay,
    CounterfactualReplayConfig,
)

SEED = 2928563038
OUT = Path(__file__).with_name("samples_P016v1.md")


def main():
    random.seed(SEED)
    lines = []
    lines.append("# Samples for counterfactual_replay (P016v1)")
    lines.append("")
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        cfg = CounterfactualReplayConfig()
        cfg.set_level(level)
        task = CounterfactualReplay(config=cfg)
        for _ in range(2):
            ex = task.generate_entry()
            lines.append("**Prompt:**")
            lines.append("")
            lines.append(task.render_prompt(ex.metadata))
            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(ex.answer)
            lines.append("")
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
