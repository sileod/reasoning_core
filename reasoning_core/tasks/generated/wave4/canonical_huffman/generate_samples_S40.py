import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s40_canonical_huffman.canonical_huffman import (
    CanonicalHuffman,
)

SEED = 1963225530


def main():
    random.seed(SEED)
    task = CanonicalHuffman()
    out = []
    for level in [0, 2, 5]:
        task.config.set_level(level)
        out.append("## Level {}\n".format(level))
        for i in range(2):
            e = task.generate_example()
            out.append("### Example {}\n".format(i + 1))
            out.append("**Prompt:**\n\n{}\n".format(task.render_prompt(e.metadata)))
            out.append("**Answer:**\n\n{}\n".format(e.answer))
        out.append("")
    path = Path(__file__).with_name("samples_S40.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
