import random
from pathlib import Path

from reasoning_core.tasks.generated.wave11.temporal_perspective.temporal_perspective import (
    TemporalPerspective,
)

SEED = 4275654395
LEVELS = [0, 2, 5]
PER_LEVEL = 2


def main():
    random.seed(SEED)
    task = TemporalPerspective()
    out = []
    for level in LEVELS:
        task.config.set_level(level)
        out.append(f"## Level {level}\n")
        for _ in range(PER_LEVEL):
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            out.append(prompt)
            out.append("")
            out.append(f"Answer:\n```\n{x.answer}\n```")
            out.append("")
    path = Path(__file__).with_name("samples_P004v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
