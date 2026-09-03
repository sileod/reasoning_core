import random
from pathlib import Path

from reasoning_core.tasks.generated.wave6.s60_dependency_batches.dependency_batches import (
    DependencyBatches,
    DependencyBatchesConfig,
)

SEED = 1613754928


def main():
    random.seed(SEED)
    task = DependencyBatches()
    out = []
    for level in (0, 2, 5):
        cfg = DependencyBatchesConfig()
        cfg.set_level(level)
        task.config = cfg
        out.append(f"## Level {level}\n")
        for _ in range(2):
            e = task.generate_example()
            prompt = task.render_prompt(e.metadata)
            out.append("Prompt:")
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append("Answer:")
            out.append("```")
            out.append(e.answer)
            out.append("```")
            out.append("")
    path = Path(__file__).with_name("samples_S60.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
