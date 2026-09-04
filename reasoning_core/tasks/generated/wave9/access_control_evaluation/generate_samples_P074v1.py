import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.access_control_policy_evaluation.access_control_policy_evaluation import (
    AccessControlEvaluation,
)

SEED = 3220316931


def main():
    random.seed(SEED)
    out = []
    task = AccessControlEvaluation()
    for level in (0, 2, 5):
        out.append(f"# Level {level}\n")
        for _ in range(2):
            task.config.set_level(level)
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append(f"Answer: {entry.answer}\n")
    path = Path(__file__).with_name("samples_P074v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
