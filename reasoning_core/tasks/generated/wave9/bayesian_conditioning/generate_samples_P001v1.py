import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.bayesian_conditioning.bayesian_conditioning import (
    BayesianConditioning,
    BayesianConfig,
)


def main():
    random.seed(1662004003)
    out = Path(__file__).with_name("samples_P001v1.md")
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        cfg = BayesianConfig()
        cfg.set_level(level)
        t = BayesianConditioning(config=cfg)
        for _ in range(2):
            ex = t.generate_example()
            prompt = t.render_prompt(ex.metadata)
            lines.append("### Prompt")
            lines.append(prompt)
            lines.append("")
            lines.append("### Answer")
            lines.append(ex.answer)
            lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
