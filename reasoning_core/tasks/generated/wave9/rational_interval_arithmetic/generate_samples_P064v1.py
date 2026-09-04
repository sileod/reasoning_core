import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.rational_interval_arithmetic.rational_interval_arithmetic import (
    RationalIntervalArithmetic,
)

random.seed(2382001662)


def main():
    task = RationalIntervalArithmetic()
    out = []
    for level in (0, 2, 5):
        cfg = RationalIntervalArithmetic.config_cls()
        cfg.set_level(level)
        task = RationalIntervalArithmetic()
        out.append(f"# Level {level}\n")
        for _ in range(2):
            ex = task.generate_example(level=level)
            out.append("Prompt:\n\n")
            out.append(ex.prompt if hasattr(ex, "prompt") else task.render_prompt(ex.metadata) + "\n")
            out.append("\nAnswer:\n\n")
            out.append(ex.answer + "\n\n---\n\n")
    path = Path(__file__).with_name("samples_P064v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
