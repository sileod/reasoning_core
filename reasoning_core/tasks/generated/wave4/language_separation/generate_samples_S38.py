import random
import sys
from pathlib import Path

seed = 1931387801
random.seed(seed)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.tasks.generated.wave4.s38_language_separation.language_separation import LanguageSeparation

out = Path(__file__).with_name("samples_S38.md")


def render_level(level, n_examples):
    task = LanguageSeparation()
    task.config.set_level(level)
    blocks = []
    for _ in range(n_examples):
        e = task.generate_example()
        blocks.append(
            "**Prompt:**\n\n" + task.render_prompt(e.metadata) +
            "\n\n**Answer:**\n\n" + e.answer + "\n"
        )
    return "\n\n".join(blocks)


parts = ["# Samples: language separation (S38)\n"]
for level, n in [(0, 2), (2, 2), (5, 2)]:
    parts.append("## Level %d\n" % level)
    parts.append(render_level(level, n))

out.write_text("\n\n".join(parts) + "\n")
print("wrote", out)
